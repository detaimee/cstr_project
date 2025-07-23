#!/usr/bin/env python3
import time
import csv
import threading
import numpy as np
from scipy.integrate import solve_ivp
from pymodbus.server import StartTcpServer
from pymodbus.device import ModbusDeviceIdentification
from pymodbus.datastore import ModbusSequentialDataBlock, ModbusSlaveContext, ModbusServerContext
import logging

# Logging 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. 공정 모델 (원본 유지)
# ==============================================================================
def cstr_rhs(t, x, q, Tc, Caf):
    Ca, T = x
    V = 100.0
    rho, Cp = 1_000.0, 0.239

    # 열 폭주 및 제어가 가능한 검증된 파라미터 세트
    E_R = 8750.0
    k0 = 7.2e10
    UA = 5.0e4
    dH = -5.5e4

    rA = k0 * np.exp(-E_R / max(T, 1)) * Ca
    dCa = q / V * (Caf - Ca) - rA

    # 발열 반응(dH < 0)이므로, 앞에 (-)를 붙여 열 발생(양수) 항으로 수정
    dT = (q / V * (350.0 - T)
          - dH / (rho * Cp) * rA
          + UA / (rho * Cp * V) * (Tc - T))

    return [dCa, dT]


def one_step(x, q, Tc, Caf, dt):
    sol = solve_ivp(cstr_rhs, [0, dt], x, args=(q, Tc, Caf), method='Radau', rtol=1e-6, atol=1e-8)
    Ca, T = sol.y[:, -1]
    return np.array([np.clip(Ca, -1, 1e5), np.clip(T, -1, 1e5)])


# ==============================================================================
# 2. Modbus 관련 설정 (수정된 매핑)
# ==============================================================================
# 레지스터 매핑 (INT16 사용):
# Holding Registers (40001+) & Input Registers (30001+):
# 0: T (온도) x100 [K*100] - Read Only
# 1: Ca (농도) x1000 [mol/m³*1000] - Read Only
# 2: q (유량) x100 [L/s*100] - Read/Write
# 3: Caf (공급 농도) x1000 [mol/m³*1000] - Read/Write
# 4: Tc (냉각제 온도) x100 [K*100] - Read/Write

MODBUS_PORT = 5020
dt_sim = 0.1
TEMP_NOISE_STD = 0.20

# 전역 변수
running = threading.Event()
context = None
slave_id = 1  # OpenPLC가 사용하는 slave ID


# ==============================================================================
# 3. 시뮬레이션 루프 (Modbus 통합) - Input Registers 추가
# ==============================================================================
def simulation_loop():
    global context
    
    # 초기 상태
    state = np.array([0.9, 310.0])  # [Ca, T]
    t0 = time.perf_counter()
    
    # 초기 설정값
    q_set = 100.0
    caf_set = 1.0
    tc_set = 300.0
    
    # CSV 로깅 설정
    with open('cstr_modbus_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'q_set', 'caf_set', 'tc_set', 'T_measured', 'Ca_actual'])
        
        while running.is_set():
            now = time.perf_counter() - t0
            
            try:
                # Modbus에서 설정값 읽기 (PLC가 쓴 INT 값)
                store = context[slave_id]
                q_set_int = store.getValues(3, 2, count=1)[0]      # Holding Register 2
                caf_set_int = store.getValues(3, 3, count=1)[0]    # Holding Register 3  
                tc_set_int = store.getValues(3, 4, count=1)[0]     # Holding Register 4
                
                # INT를 실제 값으로 변환 (스케일링)
                q_set = q_set_int / 100.0        # 10000 → 100.0 L/s
                caf_set = caf_set_int / 1000.0   # 1000 → 1.0 mol/m³
                tc_set = tc_set_int / 100.0      # 30000 → 300.0 K
                
                # 유효성 검사
                q_set = np.clip(q_set, 50, 150)
                caf_set = np.clip(caf_set, 0.5, 1.5)
                tc_set = np.clip(tc_set, 280, 320)
                
                # 🔍 디버깅 로그 추가
                if now % 5.0 < dt_sim:  # 5초마다 한번씩만 로그
                    logger.info(f"PLC 설정값 - q:{q_set:.1f}L/s, Caf:{caf_set:.3f}mol/m³, Tc:{tc_set:.1f}K")
                
            except Exception as e:
                logger.warning(f"Error reading from Modbus: {e}")
            
            # 시뮬레이션 한 스텝 실행
            state = one_step(state, q_set, tc_set, caf_set, dt_sim)
            Ca_true, T_true = state
            
            # 노이즈 추가 (실제 센서처럼)
            noise = np.random.randn() * TEMP_NOISE_STD
            T_measured = T_true + noise
            
            # Modbus 레지스터에 결과값 쓰기 (실제 값을 INT로 변환)
            try:
                store = context[slave_id]
                
                # T (온도) - Register 0 (INT로 스케일링)
                T_int = int(T_measured * 100)    # 310.5 K → 31050
                Ca_int = int(Ca_true * 1000)     # 0.9 mol/m³ → 900
                
                # 🔥 핵심 수정: Holding Registers AND Input Registers 둘 다에 쓰기!
                # Holding Registers (Function Code 3)
                store.setValues(3, 0, [T_int])   # T (온도)
                store.setValues(3, 1, [Ca_int])  # Ca (농도)
                
                # Input Registers (Function Code 4) - 추가!
                store.setValues(4, 0, [T_int])   # T (온도) - Input Register에도!
                store.setValues(4, 1, [Ca_int])  # Ca (농도) - Input Register에도!
                
                # 🔍 디버깅 로그 추가
                if now % 5.0 < dt_sim:  # 5초마다 한번씩만 로그
                    logger.info(f"측정값 업데이트 - T:{T_measured:.1f}K({T_int}), Ca:{Ca_true:.3f}mol/m³({Ca_int})")
                    logger.info(f"Holding & Input Registers 모두 업데이트 완료")
                
            except Exception as e:
                logger.error(f"Error writing to Modbus: {e}")
            
            # CSV 로깅
            writer.writerow([f'{now:.1f}', f'{q_set:.2f}', f'{caf_set:.3f}',
                           f'{tc_set:.2f}', f'{T_measured:.3f}', f'{Ca_true:.4f}'])
            f.flush()
            
            # 시뮬레이션 주기
            time.sleep(dt_sim)
    
    logger.info("Simulation loop ended")


# ==============================================================================
# 4. Modbus 서버 초기화 - Input Registers 추가
# ==============================================================================
def init_modbus_server():
    global context
    
    # 데이터 스토어 초기화 - Input Registers도 추가
    store = ModbusSlaveContext(
        di=ModbusSequentialDataBlock(0, [0]*100),    # Discrete Inputs
        co=ModbusSequentialDataBlock(0, [0]*100),    # Coils
        hr=ModbusSequentialDataBlock(0, [0]*200),    # Holding registers
        ir=ModbusSequentialDataBlock(0, [0]*200)     # Input registers (추가!)
    )
    
    # 서버 컨텍스트 생성 - slave id 1로 설정
    context = ModbusServerContext(slaves={1: store}, single=False)
    
    # 초기값 설정 (INT로 스케일링된 값)
    # Holding Registers 초기값
    store.setValues(3, 0, [31000])   # T 초기값: 310.0 * 100
    store.setValues(3, 1, [900])     # Ca 초기값: 0.9 * 1000  
    store.setValues(3, 2, [10000])   # q 초기값: 100.0 * 100
    store.setValues(3, 3, [1000])    # Caf 초기값: 1.0 * 1000
    store.setValues(3, 4, [30000])   # Tc 초기값: 300.0 * 100
    
    # Input Registers 초기값 (추가!)
    store.setValues(4, 0, [31000])   # T 초기값: 310.0 * 100
    store.setValues(4, 1, [900])     # Ca 초기값: 0.9 * 1000
    store.setValues(4, 2, [10000])   # q 초기값: 100.0 * 100
    store.setValues(4, 3, [1000])    # Caf 초기값: 1.0 * 1000
    store.setValues(4, 4, [30000])   # Tc 초기값: 300.0 * 100
    
    # 서버 정보
    identity = ModbusDeviceIdentification()
    identity.VendorName = 'CSTR Simulator'
    identity.ProductCode = 'CSTR-SIM'
    identity.VendorUrl = 'http://github.com/cstr_project'
    identity.ProductName = 'CSTR Process Simulator'
    identity.ModelName = 'CSTR Simulator v1.0'
    identity.MajorMinorRevision = '1.0.0'
    
    return context, identity


# ==============================================================================
# 5. 메인 실행
# ==============================================================================
if __name__ == '__main__':
    logger.info("Starting CSTR Simulator with Modbus Server...")
    logger.info("지원하는 레지스터: Holding Registers (3x) & Input Registers (4x)")
    
    # Modbus 서버 초기화
    context, identity = init_modbus_server()
    
    # 시뮬레이션 스레드 시작
    running.set()
    sim_thread = threading.Thread(target=simulation_loop, daemon=True)
    sim_thread.start()
    
    try:
        # Modbus TCP 서버 시작
        logger.info(f"Starting Modbus TCP Server on port {MODBUS_PORT}")
        StartTcpServer(
            context=context,
            identity=identity,
            address=("0.0.0.0", MODBUS_PORT)
        )
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        running.clear()
        sim_thread.join()
        logger.info("CSTR Simulator stopped.")
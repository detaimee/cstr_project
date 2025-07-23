#!/usr/bin/env python3
import time
import csv
import threading
import numpy as np
from scipy.integrate import solve_ivp
from pymodbus.server import StartTcpServer
from pymodbus.device import ModbusDeviceIdentification
from pymodbus.datastore import ModbusSequentialDataBlock, ModbusSlaveContext, ModbusServerContext
from pymodbus.payload import BinaryPayloadBuilder, BinaryPayloadDecoder
from pymodbus.constants import Endian
import struct
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
# 2. Modbus 관련 설정
# ==============================================================================
# 레지스터 매핑:
# Holding Registers (40001+):
# 0-1: T (온도) [K] - Read Only
# 2-3: Ca (농도) [mol/m³] - Read Only
# 4-5: q (유량) [L/s] - Read/Write
# 6-7: Caf (공급 농도) [mol/m³] - Read/Write
# 8-9: Tc (냉각제 온도) [K] - Read/Write

MODBUS_PORT = 5020
dt_sim = 0.1
TEMP_NOISE_STD = 0.20

# 전역 변수
running = threading.Event()
context = None
slave_id = 0x01


# ==============================================================================
# 3. Float32를 Modbus 레지스터로 변환하는 헬퍼 함수
# ==============================================================================
def float_to_registers(value):
    """Float32를 2개의 16비트 레지스터로 변환"""
    builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.BIG)
    builder.add_32bit_float(float(value))
    return builder.to_registers()


def registers_to_float(registers):
    """2개의 16비트 레지스터를 Float32로 변환"""
    decoder = BinaryPayloadDecoder.fromRegisters(registers, byteorder=Endian.BIG, wordorder=Endian.BIG)
    return decoder.decode_32bit_float()


# ==============================================================================
# 4. 시뮬레이션 루프 (Modbus 통합)
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
                # Modbus에서 설정값 읽기 (PLC가 쓴 값)
                store = context[slave_id].getValues(3, 4, count=6)  # q, Caf, Tc 읽기
                q_set = registers_to_float(store[0:2])
                caf_set = registers_to_float(store[2:4])
                tc_set = registers_to_float(store[4:6])
                
                # 유효성 검사
                q_set = np.clip(q_set, 50, 150)
                caf_set = np.clip(caf_set, 0.5, 1.5)
                tc_set = np.clip(tc_set, 280, 320)
                
            except Exception as e:
                logger.warning(f"Error reading from Modbus: {e}")
            
            # 시뮬레이션 한 스텝 실행
            state = one_step(state, q_set, tc_set, caf_set, dt_sim)
            Ca_true, T_true = state
            
            # 노이즈 추가 (실제 센서처럼)
            noise = np.random.randn() * TEMP_NOISE_STD
            T_measured = T_true + noise
            
            # Modbus 레지스터에 결과값 쓰기
            try:
                # T (온도) - Register 0-1
                T_regs = float_to_registers(T_measured)
                context[slave_id].setValues(3, 0, T_regs)
                
                # Ca (농도) - Register 2-3
                Ca_regs = float_to_registers(Ca_true)
                context[slave_id].setValues(3, 2, Ca_regs)
                
                # 현재 설정값도 다시 쓰기 (읽기 확인용)
                context[slave_id].setValues(3, 4, float_to_registers(q_set))
                context[slave_id].setValues(3, 6, float_to_registers(caf_set))
                context[slave_id].setValues(3, 8, float_to_registers(tc_set))
                
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
# 5. Modbus 서버 초기화
# ==============================================================================
def init_modbus_server():
    global context
    
    # 데이터 스토어 초기화 (10개 레지스터)
    store = ModbusSlaveContext(
        hr=ModbusSequentialDataBlock(0, [0]*10)  # Holding registers
    )
    context = ModbusServerContext(slaves=store, single=True)
    
    # 초기값 설정
    store.setValues(3, 4, float_to_registers(100.0))  # q 초기값
    store.setValues(3, 6, float_to_registers(1.0))    # Caf 초기값
    store.setValues(3, 8, float_to_registers(300.0))  # Tc 초기값
    
    # 서버 정보
    identity = ModbusDeviceIdentification()
    identity.VendorName = 'CSTR Simulator'
    identity.ProductCode = 'CSTR-SIM'
    identity.VendorUrl = 'http://github.com/cstr-project'
    identity.ProductName = 'CSTR Process Simulator'
    identity.ModelName = 'CSTR Simulator v1.0'
    identity.MajorMinorRevision = '1.0.0'
    
    return context, identity


# ==============================================================================
# 6. 메인 실행
# ==============================================================================
if __name__ == '__main__':
    logger.info("Starting CSTR Simulator with Modbus Server...")
    
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


# 📝 주요 변경사항:
# 제거된 부분:

# 모든 matplotlib 관련 import와 코드
# UI 초기화 및 갱신 함수들
# 슬라이더와 그래프 관련 코드

# 추가된 부분:

# Modbus TCP 서버 (포트 5020)
# 레지스터 매핑:

# Register 0-1: T (온도) - 읽기 전용
# Register 2-3: Ca (농도) - 읽기 전용
# Register 4-5: q (유량) - PLC에서 쓰기 가능
# Register 6-7: Caf (공급 농도) - PLC에서 쓰기 가능
# Register 8-9: Tc (냉각제 온도) - PLC에서 쓰기 가능


# Float32 ↔ Modbus 변환 함수
# CSV 로깅 (디버깅용 유지)

# 동작 방식:

# 시뮬레이터는 Modbus 서버로 실행
# PLC는 클라이언트로 연결해서 설정값(q, Caf, Tc) 쓰기
# PLC는 프로세스 값(T, Ca) 읽기
# 0.1초 주기로 시뮬레이션 업데이트
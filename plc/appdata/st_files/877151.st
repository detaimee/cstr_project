PROGRAM cstr_control
  VAR
    T_reactor AT %IW0 : INT;
    Ca_reactor AT %IW1 : INT;
    q_setpoint AT %QW2 : INT;
    Caf_setpoint AT %QW3 : INT;
    Tc_setpoint AT %QW4 : INT;
    T_high_alarm AT %QX0.0 : BOOL := FALSE;
    manual_mode AT %IX0.0 : BOOL := TRUE;
  END_VAR

  q_setpoint := 10000;
  Caf_setpoint := 1000;
  
  IF manual_mode = TRUE THEN
    Tc_setpoint := 30000;
    T_high_alarm := FALSE;
  ELSE
    Tc_setpoint := 29000;
    T_high_alarm := TRUE;
  END_IF;

END_PROGRAM


CONFIGURATION Config0

  RESOURCE Res0 ON PLC
    TASK task0(INTERVAL := T#100ms,PRIORITY := 0);
    PROGRAM instance0 WITH task0 : cstr_control;
  END_RESOURCE
END_CONFIGURATION
FROM zakharbz/labshock-plc:v1.1.1

# app data files 복사
COPY ./appdata/st_files /workdir/webserver/st_files
COPY ./appdata/openplc.db /workdir/webserver/openplc.db
COPY ./appdata/active_program /workdir/webserver/active_program
COPY ./appdata/mbconfig.cfg /workdir/webserver/mbconfig.cfg

# OpenPLC 시작
CMD ["./start_openplc.sh"]

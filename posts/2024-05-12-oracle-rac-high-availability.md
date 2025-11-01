---
layout: post
title: Oracle RAC - Alta Disponibilidad en Entornos Críticos
tags: [oracle, rac, cluster, high-availability, grid-infrastructure, enterprise]
---

Oracle Real Application Clusters (RAC) sigue siendo la solución de referencia para alta disponibilidad en bases de datos empresariales. En mis años implementando y manteniendo clusters RAC en entornos de producción críticos, he aprendido que el éxito no está solo en la instalación, sino en el diseño y monitoreo proactivo.

## Arquitectura RAC: Fundamentos Críticos

### 1. Componentes Core del Cluster

```sql
-- Verificación del estado del cluster
SELECT inst_id,
       instance_name,
       host_name,
       status,
       startup_time,
       version
FROM gv$instance
ORDER BY inst_id;

-- Estado de los servicios de Grid Infrastructure
SELECT name,
       state,
       target,
       type
FROM v$cluster_resource
WHERE type IN ('ora.service.type', 'ora.database.type', 'ora.listener.type')
ORDER BY type, name;

-- Verificación de interconnect
SELECT inst_id,
       name,
       ip_address,
       is_public,
       source
FROM gv$cluster_interconnects
ORDER BY inst_id, name;
```

### 2. Configuración de Storage Compartido

```bash
# Verificación ASM Diskgroups (desde Grid Infrastructure)
asmcmd lsdg

# Estado detallado de discos ASM
SELECT group_number,
       name as diskgroup_name,
       state,
       type,
       total_mb,
       free_mb,
       ROUND((free_mb/total_mb)*100, 2) as free_pct,
       required_mirror_free_mb,
       usable_file_mb
FROM v$asm_diskgroup;

# Verificación de balance automático ASM
SELECT group_number,
       operation,
       state,
       power,
       actual,
       sofar,
       est_work,
       est_rate,
       est_minutes
FROM v$asm_operation
WHERE state = 'RUN';
```

## Configuración Avanzada para Producción

### 1. Services y Load Balancing

```sql
-- Crear service optimizado para aplicaciones OLTP
BEGIN
    DBMS_SERVICE.CREATE_SERVICE(
        service_name => 'OLTP_SERVICE',
        network_name => 'OLTP_SERVICE',
        failover_method => 'BASIC',
        failover_type => 'SELECT',
        failover_retries => 10,
        failover_delay => 5,
        clb_goal => DBMS_SERVICE.CLB_GOAL_LONG,
        rlb_goal => DBMS_SERVICE.RLB_GOAL_THROUGHPUT,
        edition => 'ORA$BASE'
    );
END;
/

-- Configurar preferred/available instances
BEGIN
    DBMS_SERVICE.MODIFY_SERVICE(
        service_name => 'OLTP_SERVICE',
        preferred_instances => 'PROD1,PROD2',
        available_instances => 'PROD3,PROD4'
    );
END;
/

-- Iniciar el servicio
BEGIN
    DBMS_SERVICE.START_SERVICE('OLTP_SERVICE');
END;
/

-- Monitoreo de servicios
SELECT service_id,
       service_name,
       network_name,
       creation_date,
       failover_method,
       failover_type,
       clb_goal,
       rlb_goal
FROM dba_services
WHERE service_name NOT IN ('SYS$BACKGROUND', 'SYS$USERS');
```

### 2. Connection Load Balancing Inteligente

```sql
-- Configuración de métricas para runtime load balancing
ALTER SYSTEM SET service_names = 
'OLTP_SERVICE,BATCH_SERVICE,REPORT_SERVICE' 
SCOPE=BOTH SID='*';

-- Habilitar server-side load balancing
ALTER SYSTEM SET remote_listener = 
'(ADDRESS_LIST=
  (ADDRESS=(PROTOCOL=tcp)(HOST=scan-cluster.domain.com)(PORT=1521))
)' SCOPE=BOTH SID='*';

-- Configurar connection pooling
ALTER SYSTEM SET local_listener = 
'(ADDRESS_LIST=
  (ADDRESS=(PROTOCOL=tcp)(HOST=node1-vip.domain.com)(PORT=1521))
)' SCOPE=BOTH SID='PROD1';

-- Métricas de load balancing en tiempo real
SELECT service_name,
       inst_id,
       throughput,
       goodness,
       flags,
       delta,
       timestamp
FROM gv$servicemetric
WHERE service_name IN ('OLTP_SERVICE', 'BATCH_SERVICE')
ORDER BY service_name, inst_id;
```

### 3. Cache Fusion y Global Enqueue Services

```sql
-- Análisis de Global Cache Services (GCS)
SELECT block_class,
       gc_buffer_busy,
       gc_cr_blocks_received,
       gc_cr_blocks_served,
       gc_current_blocks_received,
       gc_current_blocks_served
FROM gv$gc_element
WHERE inst_id = 1;

-- Monitoreo de Global Enqueue Services (GES)
SELECT resource_name,
       lock_type,
       mode_held,
       mode_requested,
       state,
       blocked,
       blocker
FROM gv$ges_enqueue
WHERE blocked > 0
ORDER BY blocked DESC;

-- Estadísticas de Cache Fusion
SELECT stat_name,
       inst_id,
       value,
       ROUND(value / (SELECT value FROM gv$sysstat WHERE name = 'user commits' AND inst_id = 1), 4) as per_commit
FROM gv$sysstat
WHERE name LIKE 'gc %'
AND value > 0
ORDER BY inst_id, stat_name;
```

## Monitoreo y Diagnóstico RAC

### 1. Performance Monitoring Específico RAC

```sql
-- Vista consolidada de performance RAC
CREATE OR REPLACE VIEW v_rac_performance_summary AS
WITH instance_stats AS (
    SELECT inst_id,
           SUM(CASE WHEN name = 'user commits' THEN value END) as commits,
           SUM(CASE WHEN name = 'user rollbacks' THEN value END) as rollbacks,
           SUM(CASE WHEN name = 'execute count' THEN value END) as executions,
           SUM(CASE WHEN name = 'physical reads' THEN value END) as phys_reads,
           SUM(CASE WHEN name = 'physical writes' THEN value END) as phys_writes,
           SUM(CASE WHEN name = 'gc cr blocks received' THEN value END) as gc_cr_received,
           SUM(CASE WHEN name = 'gc current blocks received' THEN value END) as gc_current_received,
           SUM(CASE WHEN name = 'gc buffer busy waits' THEN value END) as gc_buffer_busy
    FROM gv$sysstat
    WHERE name IN ('user commits', 'user rollbacks', 'execute count', 
                   'physical reads', 'physical writes',
                   'gc cr blocks received', 'gc current blocks received', 
                   'gc buffer busy waits')
    GROUP BY inst_id
),
wait_stats AS (
    SELECT inst_id,
           SUM(CASE WHEN wait_class = 'Cluster' THEN time_waited_micro END) / 1000 as cluster_wait_ms,
           SUM(CASE WHEN wait_class = 'User I/O' THEN time_waited_micro END) / 1000 as io_wait_ms,
           SUM(CASE WHEN wait_class = 'System I/O' THEN time_waited_micro END) / 1000 as sys_io_wait_ms
    FROM gv$system_event
    WHERE wait_class IN ('Cluster', 'User I/O', 'System I/O')
    GROUP BY inst_id
)
SELECT i.instance_name,
       i.host_name,
       i.status,
       s.commits,
       s.rollbacks,
       s.executions,
       ROUND(s.phys_reads / NULLIF(s.executions, 0), 2) as reads_per_exec,
       ROUND(s.phys_writes / NULLIF(s.executions, 0), 2) as writes_per_exec,
       s.gc_cr_received,
       s.gc_current_received,
       s.gc_buffer_busy,
       w.cluster_wait_ms,
       w.io_wait_ms,
       SYSDATE as snapshot_time
FROM gv$instance i
JOIN instance_stats s ON i.inst_id = s.inst_id
JOIN wait_stats w ON i.inst_id = w.inst_id
ORDER BY i.inst_id;
```

### 2. Diagnóstico de Problemas Comunes

```sql
-- Análisis de contención entre nodos
WITH blocking_analysis AS (
    SELECT blocking_session,
           blocking_instance,
           COUNT(*) as blocked_sessions,
           MAX(seconds_in_wait) as max_wait_seconds,
           LISTAGG(DISTINCT event, ', ') WITHIN GROUP (ORDER BY event) as wait_events
    FROM gv$session
    WHERE blocking_session IS NOT NULL
    GROUP BY blocking_session, blocking_instance
)
SELECT b.blocking_session,
       b.blocking_instance,
       b.blocked_sessions,
       b.max_wait_seconds,
       b.wait_events,
       s.username,
       s.program,
       s.machine,
       s.sql_id,
       SUBSTR(sq.sql_text, 1, 100) as sql_preview
FROM blocking_analysis b
JOIN gv$session s ON b.blocking_session = s.sid 
                 AND b.blocking_instance = s.inst_id
LEFT JOIN v$sqlarea sq ON s.sql_id = sq.sql_id
ORDER BY b.blocked_sessions DESC, b.max_wait_seconds DESC;

-- Análisis de Global Cache Wait Events
SELECT event,
       inst_id,
       total_waits,
       total_timeouts,
       time_waited_micro / 1000 as time_waited_ms,
       ROUND(time_waited_micro / NULLIF(total_waits, 0) / 1000, 2) as avg_wait_ms
FROM gv$system_event
WHERE wait_class = 'Cluster'
AND total_waits > 0
ORDER BY time_waited_ms DESC;

-- Top SQL por Global Cache activity
SELECT sql_id,
       inst_id,
       executions,
       gc_buffer_busy_waits,
       gc_cr_blocks_received,
       gc_current_blocks_received,
       ROUND(gc_buffer_busy_waits / NULLIF(executions, 0), 2) as gc_busy_per_exec,
       SUBSTR(sql_text, 1, 80) as sql_preview
FROM (
    SELECT s.sql_id,
           s.inst_id,
           s.executions,
           s.gc_buffer_busy_waits,
           s.gc_cr_blocks_received,
           s.gc_current_blocks_received,
           t.sql_text,
           ROW_NUMBER() OVER (ORDER BY s.gc_buffer_busy_waits DESC) rn
    FROM gv$sql s
    JOIN v$sqlarea t ON s.sql_id = t.sql_id
    WHERE s.executions > 0
    AND s.gc_buffer_busy_waits > 100
)
WHERE rn <= 20;
```

## Procedimientos de Failover y Recovery

### 1. Automatic Failover Configuration

```sql
-- Configurar Fast Application Notification (FAN)
ALTER SYSTEM SET event = 
'10798 trace name context level 7' 
SCOPE=SPFILE SID='*';

-- Habilitar Fast Connection Failover
ALTER SYSTEM SET fast_start_mttr_target = 300 SCOPE=BOTH SID='*';
ALTER SYSTEM SET log_checkpoint_interval = 0 SCOPE=BOTH SID='*';

-- Configurar Transparent Application Failover (TAF)
-- En tnsnames.ora:
OLTP_SERVICE =
  (DESCRIPTION =
    (ADDRESS_LIST =
      (ADDRESS = (PROTOCOL = TCP)(HOST = scan-cluster)(PORT = 1521))
    )
    (CONNECT_DATA =
      (SERVER = DEDICATED)
      (SERVICE_NAME = OLTP_SERVICE)
      (FAILOVER_MODE =
        (TYPE = SELECT)
        (METHOD = BASIC)
        (RETRIES = 10)
        (DELAY = 5)
      )
    )
  )

-- Verificación de configuración TAF
SELECT machine,
       username,
       failover_type,
       failover_method,
       failed_over,
       COUNT(*) as session_count
FROM gv$session
WHERE failover_type IS NOT NULL
GROUP BY machine, username, failover_type, failover_method, failed_over;
```

### 2. Rolling Upgrade y Maintenance

```sql
-- Preparar nodo para mantenimiento
ALTER SYSTEM SET cluster_database_instances = 3 SCOPE=SPFILE SID='*';

-- Relocate services antes del shutdown
BEGIN
    DBMS_SERVICE.RELOCATE_SERVICE(
        service_name => 'OLTP_SERVICE',
        old_instance => 'PROD4',
        new_instance => 'PROD1'
    );
END;
/

-- Script para rolling restart automatizado
CREATE OR REPLACE PROCEDURE rolling_restart_cluster IS
    TYPE t_instances IS TABLE OF VARCHAR2(30);
    v_instances t_instances;
    v_sql VARCHAR2(1000);
    
BEGIN
    -- Obtener lista de instancias
    SELECT instance_name 
    BULK COLLECT INTO v_instances
    FROM gv$instance
    WHERE status = 'OPEN'
    ORDER BY inst_id;
    
    FOR i IN 1..v_instances.COUNT LOOP
        -- Relocate services
        DBMS_OUTPUT.PUT_LINE('Relocating services from ' || v_instances(i));
        
        FOR svc IN (SELECT service_name FROM dba_services 
                   WHERE service_name NOT LIKE 'SYS$%') LOOP
            BEGIN
                DBMS_SERVICE.RELOCATE_SERVICE(
                    service_name => svc.service_name,
                    old_instance => v_instances(i),
                    new_instance => v_instances(CASE WHEN i = v_instances.COUNT THEN 1 ELSE i + 1 END)
                );
            EXCEPTION
                WHEN OTHERS THEN
                    DBMS_OUTPUT.PUT_LINE('Warning: ' || SQLERRM);
            END;
        END LOOP;
        
        -- Shutdown instance (ejecutar desde srvctl externamente)
        DBMS_OUTPUT.PUT_LINE('Execute: srvctl stop instance -d PRODDB -i ' || v_instances(i));
        
        -- Wait for restart confirmation
        DBMS_OUTPUT.PUT_LINE('Execute: srvctl start instance -d PRODDB -i ' || v_instances(i));
        DBMS_OUTPUT.PUT_LINE('Press ENTER when ' || v_instances(i) || ' is back online...');
        
    END LOOP;
    
END rolling_restart_cluster;
```

## Optimización Específica RAC

### 1. Sequence Cache y Hot Blocks

```sql
-- Optimizar sequences para RAC
ALTER SEQUENCE order_seq CACHE 1000 ORDER;
ALTER SEQUENCE invoice_seq CACHE 500 NOORDER;

-- Identificar hot blocks y contención
SELECT object_name,
       object_type,
       block_type,
       blocks,
       gc_buffer_busy_waits
FROM (
    SELECT o.object_name,
           o.object_type,
           'DATA' as block_type,
           COUNT(*) as blocks,
           SUM(gc_buffer_busy_waits) as gc_buffer_busy_waits
    FROM gv$bh b
    JOIN dba_objects o ON b.objd = o.data_object_id
    WHERE gc_buffer_busy_waits > 0
    GROUP BY o.object_name, o.object_type
    ORDER BY gc_buffer_busy_waits DESC
)
WHERE ROWNUM <= 10;

-- Reverse key indexes para secuencias
CREATE INDEX idx_orders_id_reverse 
ON orders (order_id) REVERSE
TABLESPACE indexes_ts;

-- Partitioning para reducir contention
ALTER TABLE high_volume_table 
MODIFY PARTITION BY HASH (id) PARTITIONS 16;
```

### 2. Application-Side RAC Optimization

```sql
-- Configurar connection pooling óptimo
-- En aplicación Java/Spring:
-- spring.datasource.hikari.maximum-pool-size=20
-- spring.datasource.hikari.minimum-idle=5
-- oracle.jdbc.implicitStatementCacheSize=20
-- oracle.net.CONNECT_TIMEOUT=10000
-- oracle.jdbc.ReadTimeout=30000

-- Binding variables para mejor cursor sharing
-- ✅ Correcto
SELECT * FROM orders WHERE customer_id = :customer_id AND status = :status;

-- ❌ Incorrecto (genera múltiples cursores)
SELECT * FROM orders WHERE customer_id = 12345 AND status = 'PENDING';

-- Instance-aware connection routing
CREATE OR REPLACE FUNCTION get_preferred_instance(p_service_name VARCHAR2) 
RETURN VARCHAR2 IS
    v_instance VARCHAR2(30);
BEGIN
    SELECT instance_name
    INTO v_instance
    FROM (
        SELECT i.instance_name,
               s.throughput,
               ROW_NUMBER() OVER (ORDER BY s.goodness DESC, s.throughput ASC) rn
        FROM gv$instance i
        JOIN gv$servicemetric s ON i.inst_id = s.inst_id
        WHERE s.service_name = p_service_name
        AND i.status = 'OPEN'
    )
    WHERE rn = 1;
    
    RETURN v_instance;
EXCEPTION
    WHEN NO_DATA_FOUND THEN
        RETURN 'PROD1';  -- Default instance
END;
```

## Scripts de Monitoreo Automatizado

### 1. Health Check RAC Completo

```sql
-- Procedure de health check automatizado
CREATE OR REPLACE PROCEDURE rac_health_check IS
    v_report CLOB := '=== RAC CLUSTER HEALTH CHECK ===' || CHR(10);
    
BEGIN
    -- Cluster status
    v_report := v_report || CHR(10) || '--- CLUSTER STATUS ---' || CHR(10);
    FOR rec IN (
        SELECT inst_id,
               instance_name,
               host_name,
               status,
               TO_CHAR(startup_time, 'DD-MON-YY HH24:MI') as startup_time
        FROM gv$instance
        ORDER BY inst_id
    ) LOOP
        v_report := v_report || 'Instance ' || rec.inst_id || ': ' || 
                   rec.instance_name || ' (' || rec.host_name || ') - ' ||
                   rec.status || ' since ' || rec.startup_time || CHR(10);
    END LOOP;
    
    -- Services status
    v_report := v_report || CHR(10) || '--- SERVICES STATUS ---' || CHR(10);
    FOR rec IN (
        SELECT service_name,
               COUNT(DISTINCT inst_id) as active_instances,
               LISTAGG(CASE WHEN enabled = 'Y' THEN TO_CHAR(inst_id) END, ',') 
               WITHIN GROUP (ORDER BY inst_id) as enabled_on
        FROM gv$active_services
        WHERE service_name NOT LIKE 'SYS$%'
        GROUP BY service_name
        ORDER BY service_name
    ) LOOP
        v_report := v_report || rec.service_name || ': ' || 
                   rec.active_instances || ' instances [' || 
                   NVL(rec.enabled_on, 'NONE') || ']' || CHR(10);
    END LOOP;
    
    -- ASM Diskgroups
    v_report := v_report || CHR(10) || '--- ASM DISKGROUPS ---' || CHR(10);
    FOR rec IN (
        SELECT name,
               state,
               type,
               total_mb,
               free_mb,
               ROUND((free_mb/total_mb)*100, 1) as free_pct
        FROM v$asm_diskgroup
        ORDER BY name
    ) LOOP
        v_report := v_report || rec.name || ': ' || rec.state || 
                   ' (' || rec.type || ') - ' || 
                   ROUND(rec.total_mb/1024, 1) || 'GB total, ' ||
                   rec.free_pct || '% free' || CHR(10);
    END LOOP;
    
    -- Top wait events
    v_report := v_report || CHR(10) || '--- TOP CLUSTER WAIT EVENTS ---' || CHR(10);
    FOR rec IN (
        SELECT event,
               SUM(total_waits) as total_waits,
               ROUND(SUM(time_waited_micro)/1000, 2) as total_wait_ms,
               ROUND(AVG(time_waited_micro/NULLIF(total_waits,0))/1000, 2) as avg_wait_ms
        FROM gv$system_event
        WHERE wait_class = 'Cluster'
        AND total_waits > 0
        GROUP BY event
        ORDER BY SUM(time_waited_micro) DESC
        FETCH FIRST 5 ROWS ONLY
    ) LOOP
        v_report := v_report || rec.event || ': ' || 
                   rec.total_waits || ' waits, ' ||
                   rec.total_wait_ms || 'ms total (' ||
                   rec.avg_wait_ms || 'ms avg)' || CHR(10);
    END LOOP;
    
    -- Output report
    DBMS_OUTPUT.ENABLE(1000000);
    DBMS_OUTPUT.PUT_LINE(v_report);
    
END rac_health_check;
```

### 2. Alertas Automáticas RAC

```sql
-- Sistema de alertas para problemas RAC críticos
CREATE OR REPLACE PROCEDURE check_rac_alerts IS
    v_alert_threshold NUMBER := 5;  -- segundos
    v_down_instances NUMBER := 0;
    v_cluster_waits NUMBER := 0;
    
BEGIN
    -- Check for down instances
    SELECT COUNT(*)
    INTO v_down_instances
    FROM gv$instance
    WHERE status != 'OPEN';
    
    IF v_down_instances > 0 THEN
        send_alert('RAC_INSTANCE_DOWN', v_down_instances || ' instance(s) down');
    END IF;
    
    -- Check for excessive cluster waits
    SELECT COUNT(*)
    INTO v_cluster_waits
    FROM gv$session_wait
    WHERE wait_class = 'Cluster'
    AND seconds_in_wait > v_alert_threshold;
    
    IF v_cluster_waits > 10 THEN
        send_alert('RAC_CLUSTER_CONTENTION', 
                  v_cluster_waits || ' sessions waiting > ' || 
                  v_alert_threshold || ' seconds');
    END IF;
    
    -- Check service availability
    FOR rec IN (
        SELECT service_name,
               COUNT(*) as expected_instances,
               COUNT(CASE WHEN enabled = 'Y' THEN 1 END) as active_instances
        FROM gv$active_services
        WHERE service_name IN ('OLTP_SERVICE', 'BATCH_SERVICE')
        GROUP BY service_name
        HAVING COUNT(CASE WHEN enabled = 'Y' THEN 1 END) < 
               COUNT(*) * 0.5  -- Menos del 50% de instancias activas
    ) LOOP
        send_alert('RAC_SERVICE_DEGRADED',
                  rec.service_name || ': only ' || rec.active_instances || 
                  ' of ' || rec.expected_instances || ' instances active');
    END LOOP;
    
END check_rac_alerts;

-- Job para ejecutar cada 2 minutos
BEGIN
    DBMS_SCHEDULER.CREATE_JOB(
        job_name => 'RAC_HEALTH_MONITOR',
        job_type => 'STORED_PROCEDURE',
        job_action => 'check_rac_alerts',
        start_date => SYSTIMESTAMP,
        repeat_interval => 'FREQ=MINUTELY; INTERVAL=2',
        enabled => TRUE
    );
END;
```

## Mejores Prácticas de Producción

### 1. Configuración de Red y Storage

```bash
# Configuración de interconnect dedicado
# /etc/hosts optimizado para RAC
192.168.10.11  node1-priv
192.168.10.12  node2-priv
192.168.10.13  node3-priv
192.168.10.14  node4-priv

# Verificación de latencia de interconnect
ping -c 100 -i 0.01 node2-priv | tail -1

# Configuración de multipath para ASM
multipath -ll

# Optimización de parámetros de red
echo 'net.core.rmem_default = 262144' >> /etc/sysctl.conf
echo 'net.core.rmem_max = 4194304' >> /etc/sysctl.conf
echo 'net.core.wmem_default = 262144' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 1048576' >> /etc/sysctl.conf
```

### 2. Backup y Recovery en RAC

```sql
-- Configuración RMAN para RAC
CONFIGURE CHANNEL DEVICE TYPE DISK PARALLELISM 4;
CONFIGURE BACKUP OPTIMIZATION ON;
CONFIGURE RETENTION POLICY TO RECOVERY WINDOW OF 7 DAYS;

-- Backup paralelo por instancia
RUN {
    ALLOCATE CHANNEL ch1 DEVICE TYPE DISK CONNECT 'sys/password@PROD1';
    ALLOCATE CHANNEL ch2 DEVICE TYPE DISK CONNECT 'sys/password@PROD2';
    ALLOCATE CHANNEL ch3 DEVICE TYPE DISK CONNECT 'sys/password@PROD3';
    ALLOCATE CHANNEL ch4 DEVICE TYPE DISK CONNECT 'sys/password@PROD4';
    
    BACKUP DATABASE PLUS ARCHIVELOG;
    
    RELEASE CHANNEL ch1;
    RELEASE CHANNEL ch2;
    RELEASE CHANNEL ch3;
    RELEASE CHANNEL ch4;
}

-- Restore point para rollback rápido
CREATE RESTORE POINT before_upgrade GUARANTEE FLASHBACK DATABASE;

-- Recovery en RAC
RECOVER DATABASE USING BACKUP CONTROLFILE UNTIL TIME '2024-07-20 14:30:00';
ALTER DATABASE OPEN RESETLOGS;
```

## Conclusiones RAC

Oracle RAC exitoso requiere:

🏗️ **Diseño cuidadoso**: Interconnect, storage y red optimizados  
📊 **Monitoreo proactivo**: Métricas específicas RAC y alertas automáticas  
⚡ **Services inteligentes**: Load balancing y failover automático  
🔧 **Maintenance planning**: Rolling upgrades y relocación de servicios  
🛡️ **Application awareness**: Conexiones RAC-aware y cursor sharing  

La experiencia implementando clusters RAC me ha enseñado que la tecnología es solo el 50% - el otro 50% es el diseño operacional y los procedimientos de monitoreo proactivo.

---
*¿Tienes experiencia con RAC en producción? Comparte tus casos de uso y desafíos en los comentarios.*
---
layout: post
title: Oracle Database Performance Tuning - Técnicas Avanzadas
tags: [oracle, performance, tuning, sql, optimization, enterprise]
---

La optimización de bases de datos Oracle es un arte que combina conocimiento técnico profundo con experiencia práctica. Después de años optimizando sistemas críticos y desarrollando herramientas como [MergeSourceFile](https://github.com/alegorico/MergeSourceFile) para procesamiento eficiente de SQL, he compilado las técnicas más efectivas.

## Diagnóstico: El Primer Paso Crítico

### 1. AWR (Automatic Workload Repository) Analysis

```sql
-- Generar AWR Report para análisis
SELECT output 
FROM TABLE(DBMS_WORKLOAD_REPOSITORY.AWR_REPORT_TEXT(
    l_dbid => (SELECT dbid FROM v$database),
    l_inst_num => 1,
    l_bid => :begin_snap_id,    -- Snapshot inicial
    l_eid => :end_snap_id       -- Snapshot final
));

-- Identificar Top SQL statements por elapsed time
SELECT sql_id,
       executions,
       elapsed_time_per_exec / 1000000 as avg_elapsed_sec,
       cpu_time_per_exec / 1000000 as avg_cpu_sec,
       buffer_gets_per_exec,
       SUBSTR(sql_text, 1, 60) as sql_preview
FROM (
    SELECT s.sql_id,
           s.executions_delta as executions,
           s.elapsed_time_delta / NULLIF(s.executions_delta, 0) as elapsed_time_per_exec,
           s.cpu_time_delta / NULLIF(s.executions_delta, 0) as cpu_time_per_exec,
           s.buffer_gets_delta / NULLIF(s.executions_delta, 0) as buffer_gets_per_exec,
           t.sql_text,
           ROW_NUMBER() OVER (ORDER BY s.elapsed_time_delta DESC) rn
    FROM dba_hist_sqlstat s
    JOIN dba_hist_sqltext t ON s.sql_id = t.sql_id
    WHERE s.snap_id BETWEEN :begin_snap_id AND :end_snap_id
    AND s.executions_delta > 0
)
WHERE rn <= 20;
```

### 2. Real-Time Session Analysis

```sql
-- Identificar sesiones problemáticas en tiempo real
WITH active_sessions AS (
    SELECT s.sid,
           s.serial#,
           s.username,
           s.program,
           s.machine,
           s.sql_id,
           s.event,
           s.wait_class,
           s.seconds_in_wait,
           s.state,
           w.p1text,
           w.p1,
           w.p2text, 
           w.p2,
           sq.sql_text
    FROM v$session s
    LEFT JOIN v$session_wait w ON s.sid = w.sid
    LEFT JOIN v$sqlarea sq ON s.sql_id = sq.sql_id
    WHERE s.type = 'USER'
    AND s.status = 'ACTIVE'
),
blocking_sessions AS (
    SELECT blocking_session,
           COUNT(*) as blocked_count
    FROM v$session
    WHERE blocking_session IS NOT NULL
    GROUP BY blocking_session
)
SELECT a.sid,
       a.serial#,
       a.username,
       a.program,
       a.event,
       a.seconds_in_wait,
       b.blocked_count,
       CASE 
           WHEN b.blocking_session IS NOT NULL THEN 'BLOCKER'
           WHEN a.sid IN (SELECT blocking_session FROM v$session WHERE blocking_session IS NOT NULL) THEN 'BLOCKER'
           ELSE 'NORMAL'
       END as session_type,
       SUBSTR(a.sql_text, 1, 100) as current_sql
FROM active_sessions a
LEFT JOIN blocking_sessions b ON a.sid = b.blocking_session
ORDER BY b.blocked_count DESC NULLS LAST, a.seconds_in_wait DESC;
```

## Optimización de Consultas SQL

### 1. Index Strategy Avanzada

```sql
-- Análisis de uso de índices
SELECT i.index_name,
       i.table_name,
       i.uniqueness,
       i.num_rows,
       u.monitoring,
       u.used,
       u.start_monitoring,
       u.end_monitoring
FROM dba_indexes i
LEFT JOIN v$object_usage u ON i.index_name = u.index_name
WHERE i.owner = 'APP_SCHEMA'
AND i.table_name IN ('ORDERS', 'ORDER_ITEMS', 'CUSTOMERS')
ORDER BY i.table_name, i.index_name;

-- Crear índice compuesto optimizado
CREATE INDEX idx_orders_status_date_customer 
ON orders (status, order_date, customer_id)
TABLESPACE indexes_ts
COMPUTE STATISTICS
PARALLEL 4;

-- Índice funcional para consultas case-insensitive
CREATE INDEX idx_customers_upper_lastname
ON customers (UPPER(last_name))
TABLESPACE indexes_ts;

-- Índice parcial para datos activos
CREATE INDEX idx_orders_active_status
ON orders (customer_id, order_date)
WHERE status IN ('PENDING', 'PROCESSING', 'SHIPPED')
TABLESPACE indexes_ts;
```

### 2. Query Rewriting Patterns

```sql
-- Patrón 1: Eliminar subconsultas correlacionadas
-- ❌ Subconsulta correlacionada (lenta)
SELECT c.customer_id, c.name,
       (SELECT COUNT(*) 
        FROM orders o 
        WHERE o.customer_id = c.customer_id 
        AND o.order_date >= ADD_MONTHS(SYSDATE, -12)) as recent_orders
FROM customers c
WHERE c.status = 'ACTIVE';

-- ✅ JOIN optimizado (rápida)
SELECT c.customer_id, 
       c.name,
       NVL(o.recent_orders, 0) as recent_orders
FROM customers c
LEFT JOIN (
    SELECT customer_id, 
           COUNT(*) as recent_orders
    FROM orders
    WHERE order_date >= ADD_MONTHS(SYSDATE, -12)
    GROUP BY customer_id
) o ON c.customer_id = o.customer_id
WHERE c.status = 'ACTIVE';

-- Patrón 2: Window Functions para rankings
-- ✅ Ranking eficiente con analytics
SELECT customer_id,
       order_id,
       order_date,
       order_total,
       ROW_NUMBER() OVER (
           PARTITION BY customer_id 
           ORDER BY order_total DESC
       ) as order_rank,
       SUM(order_total) OVER (
           PARTITION BY customer_id
       ) as customer_total
FROM orders
WHERE order_date >= ADD_MONTHS(SYSDATE, -24);

-- Patrón 3: Optimización de EXISTS vs IN
-- ✅ EXISTS para large datasets
SELECT c.customer_id, c.name
FROM customers c
WHERE EXISTS (
    SELECT 1 
    FROM orders o
    WHERE o.customer_id = c.customer_id
    AND o.order_date >= ADD_MONTHS(SYSDATE, -6)
)
AND c.status = 'ACTIVE';
```

### 3. Hints Estratégicos y Plan Control

```sql
-- Control preciso del plan de ejecución
SELECT /*+ 
    LEADING(d c o) 
    USE_NL(c o)
    INDEX(d idx_dept_location)
    INDEX(c idx_cust_status_date)
    INDEX(o idx_orders_cust_date)
    PARALLEL(o, 4)
*/ 
d.department_name,
c.customer_name,
SUM(o.order_total) as total_sales,
COUNT(o.order_id) as order_count
FROM departments d
JOIN customers c ON d.region_id = c.region_id
JOIN orders o ON c.customer_id = o.customer_id
WHERE d.status = 'ACTIVE'
AND c.registration_date >= ADD_MONTHS(SYSDATE, -12)  
AND o.order_date >= ADD_MONTHS(SYSDATE, -3)
GROUP BY d.department_name, c.customer_name
HAVING SUM(o.order_total) > 10000
ORDER BY total_sales DESC;

-- SQL Plan Baselines para estabilidad
BEGIN
    DBMS_SPM.LOAD_PLANS_FROM_CURSOR_CACHE(
        sql_id => '9babjv8yq8ru3',
        plan_hash_value => 1234567890,
        fixed => 'YES',
        enabled => 'YES'
    );
END;
```

## Optimización de Memoria y I/O

### 1. Buffer Pool Management

```sql
-- Análisis de buffer pool efficiency
SELECT pool,
       db_block_gets,
       consistent_gets,
       physical_reads,
       ROUND((1 - (physical_reads / (db_block_gets + consistent_gets))) * 100, 2) as hit_ratio
FROM v$buffer_pool_statistics;

-- Configuración optimizada de buffer pools
ALTER SYSTEM SET db_cache_size = 4G SCOPE=BOTH;
ALTER SYSTEM SET db_keep_cache_size = 512M SCOPE=BOTH;
ALTER SYSTEM SET db_recycle_cache_size = 256M SCOPE=BOTH;

-- Asignar tablas críticas al KEEP pool
ALTER TABLE critical_lookups STORAGE (BUFFER_POOL KEEP);
ALTER TABLE transaction_log STORAGE (BUFFER_POOL RECYCLE);
```

### 2. PGA y Sort Area Tuning

```sql
-- Análisis de operaciones de sort y hash
SELECT operation,
       options,
       estimated_optimal_size / 1024 / 1024 as optimal_mb,
       estimated_onepass_size / 1024 / 1024 as onepass_mb,
       last_memory_used / 1024 / 1024 as used_mb,
       executions,
       multipasses
FROM v$sql_workarea_histogram
WHERE estimated_optimal_size > 1024 * 1024  -- > 1MB
ORDER BY estimated_optimal_size DESC;

-- Configuración PGA automática
ALTER SYSTEM SET pga_aggregate_target = 8G SCOPE=BOTH;
ALTER SYSTEM SET workarea_size_policy = AUTO SCOPE=BOTH;
ALTER SYSTEM SET "_pga_max_size" = 2G SCOPE=SPFILE;
```

### 3. I/O Optimization

```sql
-- Análisis de I/O por archivo
SELECT f.tablespace_name,
       f.file_name,
       s.phyrds as physical_reads,
       s.phywrts as physical_writes,
       s.readtim / 100 as read_time_sec,
       s.writetim / 100 as write_time_sec,
       ROUND(s.readtim / NULLIF(s.phyrds, 0), 2) as avg_read_time_ms
FROM dba_data_files f
JOIN v$filestat s ON f.file_id = s.file#
WHERE s.phyrds + s.phywrts > 1000  -- Archivos con actividad
ORDER BY s.phyrds + s.phywrts DESC;

-- Configuración async I/O
ALTER SYSTEM SET disk_asynch_io = TRUE SCOPE=SPFILE;
ALTER SYSTEM SET db_writer_processes = 4 SCOPE=SPFILE;
```

## Partitioning para Performance

### 1. Implementación de Partitioning Strategy

```sql
-- Tabla particionada por rango (fechas)
CREATE TABLE sales_data (
    sale_id NUMBER,
    sale_date DATE,
    customer_id NUMBER,
    product_id NUMBER,
    amount NUMBER(10,2),
    region_id NUMBER
)
PARTITION BY RANGE (sale_date) (
    PARTITION sales_2023_q1 VALUES LESS THAN (DATE '2023-04-01')
        TABLESPACE sales_2023_ts,
    PARTITION sales_2023_q2 VALUES LESS THAN (DATE '2023-07-01')
        TABLESPACE sales_2023_ts,
    PARTITION sales_2023_q3 VALUES LESS THAN (DATE '2023-10-01')
        TABLESPACE sales_2023_ts,
    PARTITION sales_2023_q4 VALUES LESS THAN (DATE '2024-01-01')
        TABLESPACE sales_2023_ts
);

-- Automatización de particiones nuevas
BEGIN
    DBMS_SCHEDULER.CREATE_JOB(
        job_name => 'CREATE_MONTHLY_PARTITIONS',
        job_type => 'PLSQL_BLOCK',
        job_action => '
        DECLARE
            v_partition_name VARCHAR2(30);
            v_high_value DATE;
        BEGIN
            v_high_value := ADD_MONTHS(TRUNC(SYSDATE, ''MM''), 2);
            v_partition_name := ''SALES_'' || TO_CHAR(v_high_value, ''YYYY_MM'');
            
            EXECUTE IMMEDIATE ''ALTER TABLE sales_data 
                ADD PARTITION '' || v_partition_name || 
                '' VALUES LESS THAN (DATE '''''' || 
                TO_CHAR(v_high_value, ''YYYY-MM-DD'') || '''''')'';
        END;',
        start_date => SYSTIMESTAMP,
        repeat_interval => 'FREQ=MONTHLY; BYMONTHDAY=1',
        enabled => TRUE
    );
END;
```

### 2. Partition Pruning Optimization

```sql
-- Query optimizada con partition pruning
EXPLAIN PLAN FOR
SELECT /*+ PARALLEL(s, 4) */
    region_id,
    SUM(amount) as total_sales,
    COUNT(*) as transaction_count
FROM sales_data s
WHERE sale_date >= DATE '2023-01-01'
AND sale_date < DATE '2023-04-01'    -- Partition pruning automático
AND region_id IN (1, 2, 3)
GROUP BY region_id;

SELECT * FROM TABLE(DBMS_XPLAN.DISPLAY);
```

## Monitoring y Alertas

### 1. Performance Monitoring Dashboard

```sql
-- Vista consolidada de performance metrics
CREATE OR REPLACE VIEW v_db_performance_summary AS
SELECT 
    -- CPU Usage
    (SELECT ROUND(value, 2) 
     FROM v$sysmetric 
     WHERE metric_name = 'Host CPU Utilization (%)' 
     AND ROWNUM = 1) as cpu_usage_pct,
    
    -- Memory Usage
    (SELECT ROUND(value / 1024 / 1024, 2) 
     FROM v$sysmetric 
     WHERE metric_name = 'PGA Memory Usage %' 
     AND ROWNUM = 1) as pga_usage_mb,
    
    -- I/O Performance
    (SELECT ROUND(value, 2) 
     FROM v$sysmetric 
     WHERE metric_name = 'Physical Read Total IO Requests Per Sec' 
     AND ROWNUM = 1) as read_iops,
    
    -- Session Stats
    (SELECT COUNT(*) 
     FROM v$session 
     WHERE type = 'USER' 
     AND status = 'ACTIVE') as active_sessions,
    
    -- Wait Events
    (SELECT event 
     FROM (
         SELECT event, 
                ROW_NUMBER() OVER (ORDER BY time_waited DESC) rn
         FROM v$system_event 
         WHERE wait_class != 'Idle'
     ) 
     WHERE rn = 1) as top_wait_event,
    
    SYSDATE as snapshot_time
FROM dual;
```

### 2. Automated Performance Alerts

```sql
-- Procedimiento de alertas automáticas
CREATE OR REPLACE PROCEDURE check_performance_alerts IS
    v_cpu_threshold NUMBER := 80;
    v_session_threshold NUMBER := 100;
    v_blocking_threshold NUMBER := 5;
    
    v_cpu_usage NUMBER;
    v_active_sessions NUMBER;
    v_blocking_sessions NUMBER;
    
BEGIN
    -- Check CPU usage
    SELECT value INTO v_cpu_usage
    FROM v$sysmetric 
    WHERE metric_name = 'Host CPU Utilization (%)'
    AND ROWNUM = 1;
    
    IF v_cpu_usage > v_cpu_threshold THEN
        send_alert('HIGH_CPU', 'CPU Usage: ' || v_cpu_usage || '%');
    END IF;
    
    -- Check active sessions
    SELECT COUNT(*) INTO v_active_sessions
    FROM v$session 
    WHERE type = 'USER' AND status = 'ACTIVE';
    
    IF v_active_sessions > v_session_threshold THEN
        send_alert('HIGH_SESSIONS', 'Active Sessions: ' || v_active_sessions);
    END IF;
    
    -- Check blocking sessions
    SELECT COUNT(*) INTO v_blocking_sessions
    FROM v$session
    WHERE blocking_session IS NOT NULL;
    
    IF v_blocking_sessions > v_blocking_threshold THEN
        send_alert('BLOCKING_SESSIONS', 'Blocked Sessions: ' || v_blocking_sessions);
    END IF;
    
END check_performance_alerts;

-- Job automático para ejecutar cada 5 minutos
BEGIN
    DBMS_SCHEDULER.CREATE_JOB(
        job_name => 'PERFORMANCE_MONITOR',
        job_type => 'STORED_PROCEDURE',
        job_action => 'check_performance_alerts',
        start_date => SYSTIMESTAMP,
        repeat_interval => 'FREQ=MINUTELY; INTERVAL=5',
        enabled => TRUE
    );
END;
```

## Herramientas de Automatización

### 1. Script de Health Check Automatizado

```sql
-- Health check completo de la base de datos
CREATE OR REPLACE PROCEDURE db_health_check IS
    TYPE t_results IS TABLE OF VARCHAR2(4000);
    v_results t_results := t_results();
    
BEGIN
    v_results.EXTEND; v_results(v_results.COUNT) := '=== ORACLE DATABASE HEALTH CHECK ===';
    v_results.EXTEND; v_results(v_results.COUNT) := 'Timestamp: ' || TO_CHAR(SYSDATE, 'YYYY-MM-DD HH24:MI:SS');
    
    -- Database Info
    FOR rec IN (SELECT name, created, log_mode FROM v$database) LOOP
        v_results.EXTEND; v_results(v_results.COUNT) := 'Database: ' || rec.name || ' (Created: ' || rec.created || ')';
        v_results.EXTEND; v_results(v_results.COUNT) := 'Archive Mode: ' || rec.log_mode;
    END LOOP;
    
    -- Tablespace Usage
    v_results.EXTEND; v_results(v_results.COUNT) := CHR(10) || '--- TABLESPACE USAGE ---';
    FOR rec IN (
        SELECT tablespace_name,
               ROUND(used_mb, 2) as used_mb,
               ROUND(total_mb, 2) as total_mb,
               ROUND((used_mb/total_mb)*100, 2) as pct_used
        FROM (
            SELECT ts.tablespace_name,
                   NVL(used.bytes, 0) / 1024 / 1024 as used_mb,
                   ts.bytes / 1024 / 1024 as total_mb
            FROM (SELECT tablespace_name, SUM(bytes) bytes 
                  FROM dba_data_files 
                  GROUP BY tablespace_name) ts
            LEFT JOIN (SELECT tablespace_name, SUM(bytes) bytes
                      FROM dba_segments
                      GROUP BY tablespace_name) used
            ON ts.tablespace_name = used.tablespace_name
        )
        WHERE (used_mb/total_mb)*100 > 85  -- Solo mostrar > 85%
        ORDER BY pct_used DESC
    ) LOOP
        v_results.EXTEND; 
        v_results(v_results.COUNT) := rec.tablespace_name || ': ' || 
                                     rec.used_mb || 'MB / ' || 
                                     rec.total_mb || 'MB (' || 
                                     rec.pct_used || '%)';
    END LOOP;
    
    -- Invalid Objects
    v_results.EXTEND; v_results(v_results.COUNT) := CHR(10) || '--- INVALID OBJECTS ---';
    FOR rec IN (
        SELECT owner, object_type, COUNT(*) as cnt
        FROM dba_objects
        WHERE status = 'INVALID'
        AND owner NOT IN ('SYS', 'SYSTEM', 'OUTLN')
        GROUP BY owner, object_type
        ORDER BY cnt DESC
    ) LOOP
        v_results.EXTEND; 
        v_results(v_results.COUNT) := rec.owner || '.' || rec.object_type || ': ' || rec.cnt;
    END LOOP;
    
    -- Output results
    FOR i IN 1..v_results.COUNT LOOP
        DBMS_OUTPUT.PUT_LINE(v_results(i));
    END LOOP;
    
END db_health_check;
```

## Integración con Herramientas de Desarrollo

Mi experiencia con [MergeSourceFile](https://github.com/alegorico/MergeSourceFile) ha demostrado la importancia de integrar optimización en el pipeline de desarrollo:

```sql
-- Template SQL optimizado usando MergeSourceFile
-- archivo: deploy_optimized_schema.sql

-- Variables Jinja2 para configuración por entorno
{% set tablespace_data = sql_tablespace_data %}
{% set tablespace_indexes = sql_tablespace_indexes %}
{% set parallel_degree = sql_parallel_degree | default(4) %}

-- Crear tabla optimizada con hints de performance
CREATE TABLE {{ schema_name }}.orders (
    order_id NUMBER GENERATED BY DEFAULT AS IDENTITY,
    customer_id NUMBER NOT NULL,
    order_date DATE DEFAULT SYSDATE,
    status VARCHAR2(20) DEFAULT 'PENDING',
    total_amount NUMBER(10,2),
    -- Constraints
    CONSTRAINT pk_orders PRIMARY KEY (order_id)
        USING INDEX TABLESPACE {{ tablespace_indexes }}
        STORAGE (INITIAL 10M NEXT 10M PCTINCREASE 0),
    CONSTRAINT fk_orders_customer 
        FOREIGN KEY (customer_id) 
        REFERENCES {{ schema_name }}.customers(customer_id)
)
TABLESPACE {{ tablespace_data }}
STORAGE (INITIAL 100M NEXT 100M PCTINCREASE 0)
{% if environment == "production" %}
PARALLEL {{ parallel_degree }}
{% endif %}
;

-- Índices optimizados automáticamente
{% for index_def in order_indexes %}
CREATE INDEX {{ schema_name }}.{{ index_def.name }}
ON {{ schema_name }}.orders ({{ index_def.columns }})
TABLESPACE {{ tablespace_indexes }}
{% if index_def.parallel and environment == "production" %}
PARALLEL {{ parallel_degree }}
{% endif %}
COMPUTE STATISTICS;
{% endfor %}

-- Configuración: variables.yaml
-- sql_tablespace_data: "ORDERS_DATA"
-- sql_tablespace_indexes: "ORDERS_IDX" 
-- sql_parallel_degree: 8
-- order_indexes:
--   - name: "idx_orders_customer_date"
--     columns: "customer_id, order_date"
--     parallel: true
--   - name: "idx_orders_status"
--     columns: "status"
--     parallel: false

-- Uso: msf --config deploy_config.toml
```

## Conclusiones de Performance Tuning

Las claves del tuning exitoso:

🎯 **Diagnóstico primero**: AWR, ASH, y métricas en tiempo real  
📊 **Medición constante**: Baselines y monitoring automático  
🔧 **Optimización incremental**: Cambios graduales y medibles  
⚡ **Índices inteligentes**: Estrategia basada en patrones de uso  
🗂️ **Partitioning estratégico**: Para tablas grandes y consultas temporales  
🤖 **Automatización**: Scripts y jobs para mantenimiento proactivo  

La experiencia desarrollando herramientas como `plsql_logs` y `MergeSourceFile` ha reforzado que la optimización debe estar integrada en todo el ciclo de desarrollo, no ser un afterthought.

---
*¿Has implementado alguna de estas técnicas en tu entorno? Comparte tus resultados y desafíos en los comentarios.*
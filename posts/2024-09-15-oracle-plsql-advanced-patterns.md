---
layout: post
title: Oracle PL/SQL - Patrones Avanzados para Desarrollo Empresarial
tags: [oracle, plsql, database, sql, stored-procedures, performance]
---

Oracle PL/SQL sigue siendo fundamental en entornos empresariales. En mi experiencia desarrollando el proyecto [plsql_logs](https://github.com/alegorico/plsql_logs) (logging avanzado para PL/SQL) y trabajando con sistemas críticos, he identificado patrones que marcan la diferencia entre código básico y soluciones robustas.

## Mi Experiencia con PL/SQL Logging

El proyecto **plsql_logs** implementa un sistema de logging empresarial que he desarrollado para aplicaciones críticas:

```sql
-- Ejemplo del sistema plsql_logs
CREATE OR REPLACE PACKAGE BODY pkg_logging IS
    
    PROCEDURE log_error(
        p_module     VARCHAR2,
        p_message    VARCHAR2,
        p_sql_code   NUMBER DEFAULT SQLCODE,
        p_sql_errm   VARCHAR2 DEFAULT SQLERRM
    ) IS
        PRAGMA AUTONOMOUS_TRANSACTION;
    BEGIN
        INSERT INTO application_logs (
            log_id,
            log_level,
            module_name,
            message,
            error_code,
            error_message,
            session_id,
            created_date
        ) VALUES (
            log_seq.NEXTVAL,
            'ERROR',
            p_module,
            p_message,
            p_sql_code,
            p_sql_errm,
            SYS_CONTEXT('USERENV', 'SESSIONID'),
            SYSTIMESTAMP
        );
        COMMIT;
    END log_error;

END pkg_logging;
```

## Patrones Avanzados de PL/SQL

### 1. Exception Handling Estratificado

```sql
CREATE OR REPLACE PROCEDURE process_batch_data(
    p_batch_id NUMBER,
    p_commit_size NUMBER DEFAULT 1000
) IS
    -- Variables locales
    v_processed_count NUMBER := 0;
    v_error_count NUMBER := 0;
    
    -- Exceptions definidas
    ex_batch_not_found EXCEPTION;
    ex_data_validation_error EXCEPTION;
    
    PRAGMA EXCEPTION_INIT(ex_batch_not_found, -20001);
    PRAGMA EXCEPTION_INIT(ex_data_validation_error, -20002);
    
BEGIN
    -- Logging de inicio
    pkg_logging.log_info('BATCH_PROCESSOR', 
        'Iniciando procesamiento batch ID: ' || p_batch_id);
    
    -- Validación inicial
    IF NOT batch_exists(p_batch_id) THEN
        RAISE_APPLICATION_ERROR(-20001, 'Batch no encontrado: ' || p_batch_id);
    END IF;
    
    -- Procesamiento principal con manejo granular
    FOR rec IN (
        SELECT rowid, data_column, status
        FROM batch_data 
        WHERE batch_id = p_batch_id 
        AND status = 'PENDING'
    ) LOOP
        BEGIN
            -- Procesamiento individual
            validate_and_process_record(rec.data_column);
            
            UPDATE batch_data 
            SET status = 'PROCESSED',
                processed_date = SYSTIMESTAMP
            WHERE rowid = rec.rowid;
            
            v_processed_count := v_processed_count + 1;
            
            -- Commit periódico
            IF MOD(v_processed_count, p_commit_size) = 0 THEN
                COMMIT;
                pkg_logging.log_info('BATCH_PROCESSOR', 
                    'Procesados: ' || v_processed_count || ' registros');
            END IF;
            
        EXCEPTION
            WHEN ex_data_validation_error THEN
                v_error_count := v_error_count + 1;
                
                UPDATE batch_data 
                SET status = 'ERROR',
                    error_message = SQLERRM
                WHERE rowid = rec.rowid;
                
                pkg_logging.log_error('BATCH_PROCESSOR', 
                    'Error en registro: ' || rec.rowid || ' - ' || SQLERRM);
                    
            WHEN OTHERS THEN
                v_error_count := v_error_count + 1;
                pkg_logging.log_error('BATCH_PROCESSOR', 
                    'Error inesperado: ' || SQLERRM);
                ROLLBACK;
                RAISE;
        END;
    END LOOP;
    
    COMMIT;
    
    -- Log final
    pkg_logging.log_info('BATCH_PROCESSOR', 
        'Batch completado - Procesados: ' || v_processed_count || 
        ', Errores: ' || v_error_count);

EXCEPTION
    WHEN ex_batch_not_found THEN
        pkg_logging.log_error('BATCH_PROCESSOR', 'Batch no encontrado: ' || p_batch_id);
        RAISE;
        
    WHEN OTHERS THEN
        pkg_logging.log_error('BATCH_PROCESSOR', 'Error fatal en batch: ' || SQLERRM);
        ROLLBACK;
        RAISE;
END process_batch_data;
```

### 2. Cursores Dinámicos con Performance

```sql
CREATE OR REPLACE FUNCTION generate_dynamic_report(
    p_table_name VARCHAR2,
    p_where_clause VARCHAR2 DEFAULT NULL,
    p_order_by VARCHAR2 DEFAULT NULL
) RETURN SYS_REFCURSOR IS
    
    v_cursor SYS_REFCURSOR;
    v_sql CLOB;
    v_where_clause VARCHAR2(4000);
    v_order_clause VARCHAR2(1000);
    
BEGIN
    -- Validación de seguridad
    IF NOT is_valid_table_name(p_table_name) THEN
        RAISE_APPLICATION_ERROR(-20003, 'Nombre de tabla inválido');
    END IF;
    
    -- Construcción segura de query
    v_sql := 'SELECT * FROM ' || DBMS_ASSERT.SIMPLE_SQL_NAME(p_table_name);
    
    -- WHERE clause opcional con sanitización
    IF p_where_clause IS NOT NULL THEN
        v_where_clause := sanitize_where_clause(p_where_clause);
        v_sql := v_sql || ' WHERE ' || v_where_clause;
    END IF;
    
    -- ORDER BY clause opcional
    IF p_order_by IS NOT NULL THEN
        v_order_clause := DBMS_ASSERT.ENQUOTE_NAME(p_order_by);
        v_sql := v_sql || ' ORDER BY ' || v_order_clause;
    END IF;
    
    -- Log de la query generada
    pkg_logging.log_debug('DYNAMIC_REPORT', 'SQL: ' || v_sql);
    
    -- Abrir cursor
    OPEN v_cursor FOR v_sql;
    
    RETURN v_cursor;
    
EXCEPTION
    WHEN OTHERS THEN
        pkg_logging.log_error('DYNAMIC_REPORT', 'Error generando reporte: ' || SQLERRM);
        
        IF v_cursor%ISOPEN THEN
            CLOSE v_cursor;
        END IF;
        
        RAISE;
END generate_dynamic_report;
```

### 3. Collections y Bulk Operations

```sql
CREATE OR REPLACE PROCEDURE bulk_update_salaries(
    p_department_id NUMBER,
    p_increase_percent NUMBER
) IS
    -- Types para collections
    TYPE t_employee_ids IS TABLE OF employees.employee_id%TYPE;
    TYPE t_new_salaries IS TABLE OF employees.salary%TYPE;
    
    v_employee_ids t_employee_ids;
    v_current_salaries t_new_salaries;
    v_new_salaries t_new_salaries;
    
    -- Constantes
    c_bulk_limit CONSTANT PLS_INTEGER := 1000;
    
    -- Cursor para procesamiento
    CURSOR c_employees IS
        SELECT employee_id, salary
        FROM employees
        WHERE department_id = p_department_id
        AND status = 'ACTIVE';
        
BEGIN
    pkg_logging.log_info('SALARY_UPDATE', 
        'Iniciando actualización masiva - Dept: ' || p_department_id || 
        ', Aumento: ' || p_increase_percent || '%');
    
    OPEN c_employees;
    
    LOOP
        -- Bulk fetch con límite
        FETCH c_employees 
        BULK COLLECT INTO v_employee_ids, v_current_salaries
        LIMIT c_bulk_limit;
        
        EXIT WHEN v_employee_ids.COUNT = 0;
        
        -- Calcular nuevos salarios
        v_new_salaries.DELETE;
        v_new_salaries.EXTEND(v_current_salaries.COUNT);
        
        FOR i IN 1..v_current_salaries.COUNT LOOP
            v_new_salaries(i) := v_current_salaries(i) * (1 + p_increase_percent/100);
        END LOOP;
        
        -- Bulk update
        FORALL i IN 1..v_employee_ids.COUNT
            UPDATE employees 
            SET salary = v_new_salaries(i),
                last_salary_update = SYSTIMESTAMP,
                updated_by = USER
            WHERE employee_id = v_employee_ids(i);
            
        -- Log progreso
        pkg_logging.log_info('SALARY_UPDATE', 
            'Procesados: ' || v_employee_ids.COUNT || ' empleados');
            
        -- Commit periódico
        COMMIT;
        
    END LOOP;
    
    CLOSE c_employees;
    
    pkg_logging.log_info('SALARY_UPDATE', 
        'Actualización completada - Total procesado: ' || SQL%ROWCOUNT || ' empleados');

EXCEPTION
    WHEN OTHERS THEN
        IF c_employees%ISOPEN THEN
            CLOSE c_employees;
        END IF;
        
        pkg_logging.log_error('SALARY_UPDATE', 
            'Error en actualización masiva: ' || SQLERRM);
        ROLLBACK;
        RAISE;
END bulk_update_salaries;
```

## Optimización y Performance

### 1. Hints Estratégicos

```sql
-- Query optimizada con hints específicos
SELECT /*+ LEADING(d e) USE_NL(e) INDEX(e idx_emp_dept_id) */ 
       e.employee_id,
       e.first_name,
       e.last_name,
       d.department_name,
       e.salary
FROM departments d,
     employees e
WHERE d.department_id = e.department_id
AND d.location_id = 100
AND e.hire_date >= ADD_MONTHS(SYSDATE, -36);
```

### 2. Particionamiento en PL/SQL

```sql
CREATE OR REPLACE PROCEDURE archive_old_transactions(
    p_cutoff_date DATE,
    p_partition_name VARCHAR2
) IS
    v_partition_exists NUMBER;
    v_sql VARCHAR2(4000);
    
BEGIN
    -- Verificar si la partición existe
    SELECT COUNT(*)
    INTO v_partition_exists
    FROM user_tab_partitions
    WHERE table_name = 'TRANSACTIONS_ARCHIVE'
    AND partition_name = p_partition_name;
    
    IF v_partition_exists = 0 THEN
        -- Crear nueva partición
        v_sql := 'ALTER TABLE transactions_archive 
                  ADD PARTITION ' || p_partition_name || 
                  ' VALUES LESS THAN (DATE ''' || 
                  TO_CHAR(p_cutoff_date + 1, 'YYYY-MM-DD') || ''')';
        
        EXECUTE IMMEDIATE v_sql;
        pkg_logging.log_info('ARCHIVE', 'Partición creada: ' || p_partition_name);
    END IF;
    
    -- Mover datos antiguos
    INSERT /*+ APPEND */ INTO transactions_archive
    SELECT * FROM transactions
    WHERE transaction_date < p_cutoff_date;
    
    COMMIT;
    
    -- Eliminar datos movidos
    DELETE FROM transactions
    WHERE transaction_date < p_cutoff_date;
    
    COMMIT;
    
    pkg_logging.log_info('ARCHIVE', 
        'Archivados: ' || SQL%ROWCOUNT || ' transacciones');

END archive_old_transactions;
```

## Mejores Prácticas Empresariales

### 1. Configuración de Aplicación

```sql
CREATE OR REPLACE PACKAGE pkg_config IS
    
    -- Cache de configuración en memoria
    TYPE t_config_cache IS TABLE OF VARCHAR2(4000) INDEX BY VARCHAR2(100);
    g_config_cache t_config_cache;
    g_cache_timestamp TIMESTAMP := SYSTIMESTAMP;
    g_cache_ttl INTERVAL DAY TO SECOND := INTERVAL '5' MINUTE;
    
    FUNCTION get_config_value(p_key VARCHAR2) RETURN VARCHAR2;
    PROCEDURE refresh_cache;
    PROCEDURE set_config_value(p_key VARCHAR2, p_value VARCHAR2);
    
END pkg_config;

CREATE OR REPLACE PACKAGE BODY pkg_config IS
    
    FUNCTION get_config_value(p_key VARCHAR2) RETURN VARCHAR2 IS
        v_value VARCHAR2(4000);
    BEGIN
        -- Verificar si cache necesita refresh
        IF SYSTIMESTAMP - g_cache_timestamp > g_cache_ttl THEN
            refresh_cache;
        END IF;
        
        -- Retornar valor del cache
        IF g_config_cache.EXISTS(p_key) THEN
            RETURN g_config_cache(p_key);
        ELSE
            pkg_logging.log_warning('CONFIG', 'Key no encontrada: ' || p_key);
            RETURN NULL;
        END IF;
        
    EXCEPTION
        WHEN OTHERS THEN
            pkg_logging.log_error('CONFIG', 'Error obteniendo config: ' || SQLERRM);
            RETURN NULL;
    END get_config_value;
    
    PROCEDURE refresh_cache IS
    BEGIN
        g_config_cache.DELETE;
        
        FOR rec IN (SELECT config_key, config_value FROM app_configuration) LOOP
            g_config_cache(rec.config_key) := rec.config_value;
        END LOOP;
        
        g_cache_timestamp := SYSTIMESTAMP;
        pkg_logging.log_debug('CONFIG', 'Cache refreshed - ' || g_config_cache.COUNT || ' items');
        
    EXCEPTION
        WHEN OTHERS THEN
            pkg_logging.log_error('CONFIG', 'Error refreshing cache: ' || SQLERRM);
    END refresh_cache;
    
END pkg_config;
```

## Conclusiones

El desarrollo PL/SQL empresarial exitoso requiere:

🔧 **Logging robusto**: Como demostrado en `plsql_logs`  
⚡ **Performance-first**: Bulk operations y cursores optimizados  
🛡️ **Exception handling**: Estratificado y granular  
🔒 **Seguridad**: Validación y sanitización constante  
📊 **Monitoring**: Métricas y alertas integradas  

Estos patrones han probado su valor en sistemas de producción que manejan millones de registros diarios.

---
*¿Trabajas con Oracle PL/SQL en tu empresa? Comparte tus patrones favoritos en los comentarios.*
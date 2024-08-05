-- Purpose: Backup the responses table

-- Create a job
INSERT INTO pgagent.pga_job (jobjclid, jobname, jobdesc, jobenabled, jobhostagent)
    SELECT jcl.jclid, 't_responses_backup', 'Backup the t_responses table', true, ''
    FROM pgagent.pga_jobclass jcl
    WHERE jclname='Routine Maintenance'
;

-- Create steps
-- Step 1: Delete old backup
INSERT INTO pgagent.pga_jobstep (jstjobid, jstname, jstenabled, jstkind, jstdbname, jstconnstr, jstcode, jstonerror)
    SELECT j.jobid, 'Delete old backup', true, 's', 'mthesisdb', '', 'DROP TABLE IF EXISTS t_responses_backup;', 'f'
    FROM pgagent.pga_job j WHERE j.jobname='t_responses_backup'
;

-- Step 2: Backup the responses table
INSERT INTO pgagent.pga_jobstep (jstjobid, jstname, jstenabled, jstkind, jstdbname, jstconnstr, jstcode, jstonerror)
    SELECT j.jobid, 'Backup the responses table', true, 's', 'mthesisdb', '', 'CREATE TABLE t_responses_backup AS TABLE t_responses;', 'f'
    FROM pgagent.pga_job j WHERE j.jobname='t_responses_backup'
;

-- Schedule the job to run daily at 08:45 o'clock
INSERT INTO pgagent.pga_schedule (  jscjobid, jscname, jscdesc, jscenabled, jscstart, jscend,
                                    jscminutes,
                                    jschours,
                                    jscweekdays,
                                    jscmonthdays,
                                    jscmonths)
    SELECT  j.jobid, 'Daily at 08:45', '', true, '2024-08-01 00:00:00', NULL,
            '{f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f}',
            '{f,f,f,f,f,f,f,f,t,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f}',
            '{t,t,t,t,t,t,t}',
            '{t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t,t}',
            '{t,t,t,t,t,t,t,t,t,t,t,t}'
    FROM pgagent.pga_job j WHERE j.jobname='t_responses_backup'
;

begin;

set database_name = 'TMLS';
set schema_name = 'CFPB';
set role_name = '<insert_role>';

-- create database & schema
create database if not exists identifier($database_name);

-- grant access
grant CREATE SCHEMA, MONITOR, USAGE
on database identifier($database_name)
to role identifier($role_name);

use database identifier($database_name);

create schema if not exists identifier($schema_name);

grant all privileges
on schema identifier($schema_name)
to role identifier($role_name);

use schema identifier($schema_name);

create or replace TABLE MICRO_CONSUMER_COMPLAINTS (
    ID NUMBER(38,0),
    PRODUCT VARCHAR(16777216),
    TEXT VARCHAR(16777216),
    LABEL NUMBER(38,0)
);

create or replace TABLE SAMPLE_CONSUMER_COMPLAINTS (
    ID NUMBER(38,0),
    PRODUCT VARCHAR(16777216),
    TEXT VARCHAR(16777216),
    LABEL NUMBER(38,0)
);

commit;


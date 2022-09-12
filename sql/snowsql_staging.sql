USE database TMLS;

CREATE STAGE IF NOT EXISTS tmls_workshop;

PUT file:///path/to/file/micro_consumer_complaints.csv @tmls_workshop;
PUT file:///path/to/file/sample_consumer_complaints.csv @tmls_workshop;

COPY INTO TMLS.CFPB.MICRO_CONSUMER_COMPLAINTS
FROM @tmls_workshop files = ('micro_consumer_complaints.csv.gz')
file_format = (type = CSV skip_header = 1, FIELD_OPTIONALLY_ENCLOSED_BY='"')
on_error = 'CONTINUE';

COPY INTO TMLS.CFPB.SAMPLE_CONSUMER_COMPLAINTS
FROM @tmls_workshop files = ('sample_consumer_complaints.csv.gz')
file_format = (type = CSV skip_header = 1, FIELD_OPTIONALLY_ENCLOSED_BY='"')
on_error = 'CONTINUE';

SELECT * FROM TMLS.CFPB.MICRO_CONSUMER_COMPLAINTS LIMIT 1;
SELECT * FROM TMLS.CFPB.SAMPLE_CONSUMER_COMPLAINTS LIMIT 1;

DROP STAGE IF EXISTS tmls_workshop;
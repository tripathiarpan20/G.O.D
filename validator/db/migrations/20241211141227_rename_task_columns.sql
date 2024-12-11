-- migrate:up
ALTER TABLE public.tasks RENAME COLUMN system TO field_system;
ALTER TABLE public.tasks RENAME COLUMN instruction TO field_instruction;
ALTER TABLE public.tasks RENAME COLUMN input TO field_input;
ALTER TABLE public.tasks RENAME COLUMN output TO field_output;
ALTER TABLE public.tasks RENAME COLUMN delay_times TO times_delayed;

ALTER TABLE public.tasks RENAME COLUMN end_timestamp TO termination_at;
ALTER TABLE public.tasks RENAME COLUMN created_timestamp TO created_at;
ALTER TABLE public.tasks RENAME COLUMN delay_timestamp TO next_delay_at;
ALTER TABLE public.tasks RENAME COLUMN updated_timestamp TO updated_at;
ALTER TABLE public.tasks RENAME COLUMN started_timestamp TO started_at;
ALTER TABLE public.tasks RENAME COLUMN completed_timestamp TO completed_at;

ALTER TABLE public.tasks
    ADD COLUMN system_format text,
    ADD COLUMN trained_model_repository text;

-- Since we only accept nulls for now
ALTER TABLE public.tasks
    ALTER COLUMN no_input_format DROP NOT NULL;

ALTER TABLE public.tasks
    ALTER COLUMN format DROP NOT NULL;

-- migrate:down
ALTER TABLE public.tasks RENAME COLUMN field_system TO system;
ALTER TABLE public.tasks RENAME COLUMN field_instruction TO instruction;
ALTER TABLE public.tasks RENAME COLUMN field_input TO input;
ALTER TABLE public.tasks RENAME COLUMN field_output TO output;
ALTER TABLE public.tasks RENAME COLUMN times_delayed TO delay_times;

ALTER TABLE public.tasks RENAME COLUMN termination_at TO end_timestamp;
ALTER TABLE public.tasks RENAME COLUMN created_at TO created_timestamp;
ALTER TABLE public.tasks RENAME COLUMN next_delay_at TO delay_timestamp;
ALTER TABLE public.tasks RENAME COLUMN updated_at TO updated_timestamp;
ALTER TABLE public.tasks RENAME COLUMN started_at TO started_timestamp;
ALTER TABLE public.tasks RENAME COLUMN completed_at TO completed_timestamp;

ALTER TABLE public.tasks
    DROP COLUMN system_format,
    DROP COLUMN trained_model_repository;

ALTER TABLE public.tasks
    ALTER COLUMN no_input_format SET NOT NULL;

ALTER TABLE public.tasks
    ALTER COLUMN format SET NOT NULL;

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
    ALTER COLUMN no_input_format DROP NOT NULL,

ALTER TABLE public.tasks
    ALTER COLUMN format DROP NOT NULL,



-- migrate:down

-- migrate:down
ALTER TABLE public.tasks
    ALTER COLUMN task_id SET NOT NULL,
    ALTER COLUMN field_system SET NOT NULL,
    ALTER COLUMN field_input SET NOT NULL,
    ALTER COLUMN field_output SET NOT NULL,
    ALTER COLUMN format SET NOT NULL,
    ALTER COLUMN no_input_format SET NOT NULL,
    ALTER COLUMN test_data SET NOT NULL,
    ALTER COLUMN synthetic_data SET NOT NULL,
    ALTER COLUMN training_data SET NOT NULL,
    ALTER COLUMN miner_scores SET NOT NULL,
    ALTER COLUMN end_timestamp SET NOT NULL,
    ALTER COLUMN user_id SET NOT NULL;

ALTER TABLE public.tasks
    DROP COLUMN system_format,
    DROP COLUMN assigned_miners,
    DROP COLUMN trained_model_repository;

ALTER TABLE public.tasks RENAME COLUMN field_system TO system;
ALTER TABLE public.tasks RENAME COLUMN field_instruction TO instruction;
ALTER TABLE public.tasks RENAME COLUMN field_input TO input;
ALTER TABLE public.tasks RENAME COLUMN field_output TO output;

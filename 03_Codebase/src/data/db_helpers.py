from datetime import datetime
import getpass
import os
import pandas as pd
import platform
import psycopg2
from psycopg2.extensions import connection, cursor
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from typing import Any, Dict, List, Literal, Tuple
import yaml


class Database:
    def __init__(
        self,
        db: str = "mthesisdb",
        host: str = "193.196.52.142",
        user: str = "postgres",
        password: str = "",
        port: str = "5433",
    ) -> None:
        """Initialize database connection
        Parameters:
        db: str
            Database name
        host: str
            Host of the database
        user: str
            User of the database
        password: str
            Password of the user
        port: str
            Port of the database
        """
        self.db: str = db
        self.host: str = host
        self.user: str = user
        self.current_script_directory: str = os.path.dirname(os.path.realpath(__file__))
        # Ask for password in terminal if not provided
        if password == "":
            path: str = "../../res/keys/db.txt"
            if os.path.exists(os.path.join(self.current_script_directory, path)):
                with open(os.path.join(self.current_script_directory, path), "r") as f:
                    self.password = f.read().strip()
            else:
                self.password: str = getpass.getpass(
                    f"Enter password for user ‘{user}’: "
                )
        else:
            self.password: str = password
        self.port: str = port

    def connect(self) -> Tuple[connection, cursor, Engine]:
        """Connect to PostgreSQL database via psycopg2
        Outputs:
        self.conn: connection to the database
        self.cur: cursor of the connection
        self.engine: engine of the connection
        """
        # Connect to PostgreSQL database via psycopg2
        self.conn: connection = psycopg2.connect(
            database=self.db,
            host=self.host,
            user=self.user,
            password=self.password,
            port=self.port,
        )
        self.cur: cursor = self.conn.cursor()

        # Connect via sqlalchemy
        self.engine: Engine = create_engine(
            f"postgresql+psycopg2://{self.user}:{self.password}@[{self.host}]:{self.port}/{self.db}"
        )

        print("\033[1m\033[92mSuccessfully connected to database.\033[0m")
        return self.conn, self.cur, self.engine

    def disconnect(self) -> None:
        "Disconnect from the database"
        self.cur.close()
        self.conn.close()
        self.engine.dispose()
        print("\033[1m\033[92mSuccessfully disconnected from database.\033[0m")

    def execute_sql(self, script: str = "", sql: str = "", commit: bool = True):
        """Execute SQL query
        Parameters:
        sql: str
            SQL query to execute
        commit: bool
            Whether to commit the changes
        """
        if script != "":
            with open(script, "r") as f:
                sql_final: str = f.read()
        else:
            sql_final: str = sql

        # Execute the SQL query
        self.cur.execute(sql_final)
        if commit:
            self.conn.commit()
        print("\033[1m\033[92mSuccessfully executed SQL query.\033[0m")

    def insert_data(
        self,
        table: str,
        data: pd.DataFrame = pd.DataFrame(),
        if_exists: Literal["fail", "replace", "append"] = "append",
        updated_at: bool = False,
    ) -> None:
        """Insert data into a table
        Parameters:
        table: str
            Table to insert the data into
        data: pd.DataFrame
            Data to insert
        if_exists: str
            What to do if the table already exists
        updated_at: bool
            Whether to add the current date and time to the data
        """
        assert table != "", "Please provide the table name to insert the data into."
        # Check if a yaml file for the table exists, otherwise we need data as input
        path: str = f"../../res/db_objects_content/{table}.yaml"
        total_path: str = os.path.join(self.current_script_directory, path)
        if not os.path.exists(os.path.join(total_path)):
            assert not data.empty, "Data is empty."
        else:
            # Load the yaml file and save as df
            with open(total_path, "r") as f:
                table_content: dict = yaml.safe_load(f)
            data = pd.DataFrame(table_content)

        # Add current date and time to the data
        if updated_at:
            data["updated_at"] = pd.to_datetime("today")

        # Insert the data
        data.to_sql(table, self.engine, if_exists=if_exists, index=False)
        print(
            "\033[1m\033[92mSuccessfully inserted data into table {}.\033[0m".format(
                table
            )
        )

    def fetch_data(self, total_object: str = "", sql: str = "") -> pd.DataFrame:
        """Fetch data from a database object
        Parameters:
        total_object: str
            Name of the object to fetch data from
        sql: str
            SQL query to fetch the data
        Returns:
        pd.DataFrame
            Data fetched from the database
        """
        assert (
            total_object != "" or sql != ""
        ), "Please provide either the object name or the SQL script to fetch the data."

        if total_object != "":
            sql_final: str = f"SELECT * FROM {total_object}"
        else:
            sql_final: str = sql

        # Fetch the data
        return pd.read_sql(sql_final, self.engine)

    def fetch_next_experiments(
        self,
        top: int = 0,
        n_loops: int = 100,
        add_to_currentlyrunning: bool = False,
        check_system: bool = True,
    ) -> pd.DataFrame:
        """Fetch the next experiment to run
        Parameters:
        top: int
            Number of experiments to fetch
        n_loops: int
            Number of loops an experiment should run
        add_to_tcurrentlyrunning: bool
            Whether to add the experiment to t_currently_running
        Returns:
        pd.DataFrame
            Data of the next experiment to run
        """
        # Get the current system (Mac or Linux(cluster)))
        system: str = platform.system()
        if system != "Darwin" or not check_system:
            # Make sure that the experiment_id is not in t_currently_running
            sql: str = f""" SELECT v.*
                    FROM        v_experiments v
                    LEFT JOIN   t_currently_running t USING (experiment_id)
                    WHERE       v.correct_ran_loops <= {n_loops}
                                AND t.experiment_id IS NULL
                    ORDER BY    v.system DESC,
                                v.scenario ASC,
                                v.correct_ran_loops ASC
                    {"LIMIT " + str(top) if top > 0 else ""}
                """
        else:  # Only experiment runs with smaller models on Mac (Darwin)
            sql: str = f""" SELECT v.*
                    FROM        v_experiments v
                    LEFT JOIN   t_currently_running t USING (experiment_id)
                    WHERE       v.system = 'All'
                                AND v.correct_ran_loops <= {n_loops}
                                AND t.experiment_id IS NULL
                    ORDER BY    v.scenario ASC,
                                v.correct_ran_loops ASC
                    {"LIMIT " + str(top) if top > 0 else ""}
                """

        # Fetch the next experiment to run
        experiment: pd.DataFrame = pd.read_sql(sql, self.engine)
        # Check if an experiment was fetched
        if experiment.empty:
            print("No experiments are available to run.")
            return experiment

        # Add the experiment to t_currently_running
        if add_to_currentlyrunning:
            currently_running: pd.DataFrame = experiment.loc[:, ["experiment_id"]]
            currently_running["system"] = system
            self.insert_data(
                table="t_currently_running", data=currently_running, updated_at=True
            )

        return experiment

    def delete_data(
        self,
        total_object: str = "",
        sql: str = "",
        commit: bool = True,
        definitely_delete: bool = False,
    ) -> None:
        """Delete data from a table
        Parameters:
        total_object: str
            Name of the table to delete data from
        sql: str
            SQL query to delete the data
        commit: bool
            Whether to commit the changes
        definitely_delete: bool
            Whether to skip the confirmation
        """
        assert (
            total_object != "" or sql != ""
        ), "Please provide either the table name or the SQL script to delete the data."

        # Delete all data data
        if sql == "":
            sql_final: str = f"DELETE FROM {total_object}"
        else:
            sql_final: str = sql

        # Always ask for confirmation before deleting data
        if not definitely_delete:
            if (
                input(
                    f"Are you sure you want to delete the data from {total_object}? (y/n) "
                ).lower()
                == "y"
            ):
                self.execute_sql(sql=sql_final, commit=commit)
                print(
                    "\033[1m\033[92mSuccessfully deleted data from table {}.\033[0m".format(
                        total_object
                    )
                )
        elif definitely_delete:
            self.execute_sql(sql=sql_final, commit=commit)
            print(
                "\033[1m\033[92mSuccessfully deleted data from table {}.\033[0m".format(
                    total_object
                )
            )

    def create_object(
        self,
        object: str = "",
        sql: str = "",
        commit: bool = True,
        drop_if_exists: bool = False,
    ) -> None:
        """Create a database object
        Parameters:
        object: str
            Name of the object to create
        sql: str
            SQL query to create the object
        commit: bool
            Whether to commit the changes
        drop_if_exists: bool
            Whether to drop the object if it already exists
        """
        assert (
            object != "" or sql != ""
        ), "Please provide either the object name or the SQL script to create the object."

        # Drop the table if it exists
        if drop_if_exists:
            self.execute_sql(sql=f"DROP TABLE IF EXISTS {object}", commit=commit)

        # Get the stored sql script to create the table
        if sql == "":
            current_dir: str = os.path.dirname(os.path.abspath(__file__))
            file_path: str = os.path.join(
                current_dir, f"../../res/db_objects/{object}.sql"
            )
            with open(file_path, "r") as file:
                sql_final: str = file.read()
        else:
            print(
                "\033[1m\033[91mWARNING: Please save the object creation script as a sql file under res/db_objects.\033[0m"
            )
            sql_final: str = sql

        # Execute the SQL query
        self.execute_sql(sql=sql_final, commit=commit)
        print(
            "\033[1m\033[92mSuccessfully created object{}.\033[0m".format(" " + object)
        )

    def drop_object(
        self, object: str, sql: str = "", commit: bool = True, cascade: bool = False
    ) -> None:
        """Drop a database object
        Parameters:
        object: str
            Name of the object to drop
        sql: str
            SQL query to drop the object
        commit: bool
            Whether to commit the changes
        cascade: bool
            Whether to drop the object and all its dependencies
        """
        assert (
            object != "" or sql != ""
        ), "Please provide either the object name or the SQL script to drop the object."

        # SQL query to drop the table/view (cascade if necessary)
        sql_final: str = sql
        if sql == "":
            object_type = object.split("_")[0]  # Get the type of the object
            drop_type = (
                "TABLE"
                if object_type == "t"
                else "VIEW"
                if object_type == "v"
                else None
            )
            if drop_type:
                sql_final = f"DROP {drop_type} {object}"
                if cascade:
                    sql_final += " CASCADE"

        # Execute the SQL query
        # Always ask in the terminal if you are sure
        if input(f"Are you sure you want to drop {object}? (y/n): ").lower() == "y":
            self.execute_sql(sql=sql_final, commit=commit)
            print(
                "\033[1m\033[92mSuccessfully dropped object {}.\033[0m".format(object)
            )

    def update_data(
        self,
        object: str,
        data: pd.DataFrame,
        update_cols: List[str],
        commit: bool = True,
    ) -> None:
        """Update data in an object
        Parameters:
        object: str
            Name of the object to update data in
        data: pd.DataFrame
            Data to update in the object
        update_cols: List[str]
            Columns to update in the object
        commit: bool
            Whether to commit the changes
        """
        # Add updated_at column and to update_cols
        data["updated_at"] = pd.to_datetime("today")
        update_cols.append("updated_at")

        # Update row for row
        for _, row in data.iterrows():
            # Try to insert rows, if they exist, update them
            try:
                self.insert_data(table=object, data=row.to_frame().T, updated_at=False)
            except Exception as e:
                print(f"Error: {e}")
                print("The primary key already exists, the data will be updated.")

                # Get the columns to update the data
                update_cols_str: str = ", ".join(
                    [f"{col} = '{row[col]}'" for col in update_cols]
                )

                # Get the columns to filter the data
                # The where columns are all columns in data without the update_cols
                where_cols: List[str] = [
                    col for col in data.columns if col not in update_cols
                ]
                where_cols_str: str = "AND ".join(
                    [f"{col} = '{row[col]}'" for col in where_cols]
                )

                sql: str = (
                    f"UPDATE {object} SET {update_cols_str} WHERE {where_cols_str};"
                )
                self.execute_sql(sql=sql, commit=commit)

        # Done
        print(
            "\033[1m\033[92mSuccessfully inserted/updated data in table {}.\033[0m".format(
                object
            )
        )

    def update_master_data(
        self,
        table: str,
        columns_to_update: List[str],
        columns_to_filter: Dict[str, List[Any]],
        updated_at: bool = True,
    ) -> None:
        "Function to quickly update certain columns in the master data tables"
        # Get master data from yaml file
        print(f"{datetime.now()} | Getting master data from yaml file.")
        path: str = f"../../res/db_objects_content/{table}.yaml"
        total_path: str = os.path.join(self.current_script_directory, path)
        if os.path.exists(os.path.join(total_path)):
            # Load the yaml file and save as df
            with open(total_path, "r") as f:
                table_content: dict = yaml.safe_load(f)
            data: pd.DataFrame = pd.DataFrame(table_content)
        else:
            raise FileNotFoundError("No yaml file found for the table.")

        # Add current date and time to the data
        if updated_at:
            data["updated_at"] = pd.to_datetime("today")
            columns_to_update.append("updated_at")

        # Filter for the data that needs to be updated
        for col, values in columns_to_filter.items():
            data = data.loc[data[col].isin(values)]

        # Update the data
        print(f"{datetime.now()} | Updating the data in the database.")
        for _, row in data.iterrows():
            sql = f"""
            UPDATE {table}
            SET {", ".join([f"{col} = '{row[col]}'" for col in columns_to_update])}
            WHERE {", ".join([f"{col} = '{row[col]}'" for col in columns_to_filter.keys()])}
            ;
            """
            self.execute_sql(sql=sql, commit=True)

    def add_scenario_1_persona(self) -> None:
        "Use the bias master data of scenario 0_normal to generate 1_persona"
        "The only changes are bias_id=bias_id+10 and scenario=1_persona"
        sql: str = """
            INSERT INTO t_biases (bias_id, bias, experiment_type, scenario, content, variables, response_type, target_response, part, parts_total)
            SELECT 
                bias_id + 10 AS bias_id, 
                bias, 
                experiment_type, 
                '1_persona' AS scenario, 
                content, 
                variables, 
                response_type, 
                target_response, 
                part, 
                parts_total
            FROM 
                t_biases
            WHERE
                scenario = '0_normal';
        """
        self.execute_sql(sql=sql, commit=True)

    def add_model_temperatures(self) -> None:
        "Use the model master data with temperature 0.7 to generate the other temperatures"
        "The only changes is temperature"
        temperatures: List[str] = ["1", "1.3"]
        total_sql: str = ""
        for i, temperature in enumerate(temperatures):
            sql: str = f"""
                INSERT INTO t_models (model_id, model, local, temperature, system, release_date, last_updated_at, download_date, size, number_parameters, model_architecture, ollama_id)
                SELECT
                    model_id + {i+1} AS model_id,
                    model,
                    local,
                    '{temperature}' AS temperature,
                    system,
                    release_date,
                    last_updated_at,
                    download_date,
                    size,
                    number_parameters,
                    model_architecture,
                    ollama_id
                FROM 
                    t_models
                WHERE
                    temperature = '0.7';
            """
            total_sql += sql
        self.execute_sql(sql=total_sql, commit=True)
        print(
            "\033[1m\033[92mSuccessfully added models with other temperatures.\033[0m"
        )

    def cleanup_responses(self) -> None:
        "Cleanup the responses table from wrong formatted answers"
        # Delete all responses that are not A, B, C, D or a number from 0 to 1000
        # First select the data and export it for inspection
        sql = r"""
            FROM t_responses
            WHERE
                (response_type = 'choice' AND response NOT IN ('A', 'B', 'Failed prompt'))
                OR (response_type = 'numerical' AND response NOT SIMILAR TO '([0-9]{1,3}|1000)(\.[0-9]{1,2})?' AND response NOT IN ('Failed prompt'))
            ;
        """

        data: pd.DataFrame = self.fetch_data(sql=f"SELECT * {sql}")
        data.to_excel("responses_to_check.xlsx", index=False)  # Export as excel
        count_prior: int = len(data)

        if (
            input(
                "Please inspect the data in responses_to_check.xlsx before continuing. Type 'y' to continue and 'n' to abort: "
            )
            == "y"
        ):
            self.delete_data(sql=f"DELETE {sql}", commit=True)
            data = self.fetch_data(sql=f"SELECT * {sql}")
            count_after: int = len(data)
            print(
                f"{datetime.now()} | Deleted {count_prior - count_after} unusable responses from the responses table."
            )
        # Delete excel
        os.remove("responses_to_check.xlsx")


if __name__ == "__main__":
    import inquirer
    import os
    import sys

    # Add total codebase
    parent_dir: str = os.path.dirname(os.path.realpath(__file__ + "../../"))
    sys.path.append(parent_dir)

    # Connect to the database
    db = Database()
    db.connect()

    # Get all function names from the database class
    functions: List[str] = [
        func
        for func in dir(db)
        if callable(getattr(db, func)) and not func.startswith("__")
    ]
    functions.remove("connect")
    functions.remove("disconnect")
    functions.sort()

    # Ask the user which function to run
    questions: List = [
        inquirer.List(
            name="function",
            message="Which function do you want to run?",
            choices=functions,
        )
    ]
    answers: dict | None = inquirer.prompt(questions)

    additional_answers: dict | None = {}  # Initialize additional answers
    if answers is None:
        print("No function selected.")
        sys.exit()

    # If function is create_object, get all object names in res/db_objects
    if (answers["function"] == "create_object") or (
        answers["function"] == "drop_object"
    ):
        # Get all object names
        objects: List[str] = [
            obj.replace(".sql", "")
            for obj in os.listdir(os.path.join(parent_dir, "../res/db_objects"))
        ]
        objects.sort()

        questions: List = [
            inquirer.List(
                name="object",
                message=f"Which object do you want to {answers['function'].replace('_object', '')}?",
                choices=objects,
            )
        ]
        additional_answers = inquirer.prompt(questions)

    if answers["function"] == "fetch_data":
        # Get all object names
        objects: List[str] = [
            obj.replace(".sql", "")
            for obj in os.listdir(os.path.join(parent_dir, "../res/db_objects"))
        ]
        objects.sort()

        questions: List = [
            inquirer.List(
                name="total_or_sql",
                message="Do you want to select all the data of an object or write a specific SQL query?",
                choices=["Select total object", "Write SQL query"],
            )
        ]
        total_or_sql: dict | None = inquirer.prompt(questions)

        if total_or_sql is None:
            print("No choice made.")
            sys.exit()
        elif total_or_sql["total_or_sql"] == "Select total object":
            questions: List = [
                inquirer.List(
                    name="total_object",
                    message="Which object do you want to select all the data from?",
                    choices=objects,
                )
            ]
            additional_answers = inquirer.prompt(questions)
        elif total_or_sql["total_or_sql"] == "Write SQL query":
            questions: List = [
                inquirer.Editor(
                    "sql", message="Provide the sql query to fetch the data"
                )
            ]
            additional_answers = inquirer.prompt(questions)

    # Run the function
    print(getattr(db, answers["function"])(**additional_answers))

    # db.update_master_data(table="t_biases", columns_to_update=["content"], columns_to_filter={"bias_id": [601, 602]})
    # sql = "DELETE FROM t_responses WHERE bias_id in (601, 602)"
    # db.delete_data(sql=sql, commit=True)
    # db.delete_data(total_object="t_currently_running")
    # db.cleanup_responses()
    db.disconnect()

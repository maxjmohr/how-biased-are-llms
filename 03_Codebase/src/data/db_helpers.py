import getpass
import os
import pandas as pd
import psycopg2
from psycopg2.extensions import connection, cursor
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from typing import List, Literal, Tuple
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

    def delete_data(
        self, total_object: str, sql: str = "", commit: bool = True
    ) -> None:
        """Delete data from a table
        Parameters:
        total_object: str
            Name of the table to delete data from
        sql: str
            SQL query to delete the data
        commit: bool
            Whether to commit the changes
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
        if input("Are you sure you want to delete the data? (y/n) ").lower() == "y":
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

    def drop_object(self, object: str, sql: str = "", commit: bool = True) -> None:
        """Drop a database object
        Parameters:
        object: str
            Name of the object to drop
        sql: str
            SQL query to drop the object
        commit: bool
            Whether to commit the changes
        """
        assert (
            object != "" or sql != ""
        ), "Please provide either the object name or the SQL script to drop the object."

        # SQL query to drop the table
        object_type = object.split("_")[0]  # Get the type of the object
        if sql == "" and object_type == "t":
            sql_final: str = f"DROP TABLE {object} CASCADE"
        elif sql == "" and object_type == "v":
            sql_final: str = f"DROP VIEW {object}"
        else:
            sql_final: str = sql

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

        # Get the columns to update the data
        update_cols_str: str = ", ".join(
            [f"{col} = '{data[col]}'" for col in update_cols]
        )

        # Get the columns to filter the data
        # The where columns are all columns in data without the update_cols
        where_cols: List[str] = [col for col in data.columns if col not in update_cols]
        where_cols_str: str = ", ".join(
            [f"{col} = '{data[col]}'" for col in where_cols]
        )

        # Update row for row
        for i in range(len(data)):
            # Try to update for rows that exist
            try:
                sql: str = (
                    f"UPDATE {object} SET {update_cols_str} WHERE {where_cols_str};"
                )
                self.execute_sql(sql=sql, commit=commit)
                print(
                    "\033[1m\033[92mSuccessfully updated data in table {}.\033[0m".format(
                        object
                    )
                )
            # If the line does not exist, insert it
            except Exception as e:
                print(f"Error: {e}")
                print("The row does not exist yet, it will be inserted.")
                sql: str = f"""
                    INSERT INTO {object} ({", ".join(data.columns)})
                    VALUES ({", ".join([f"'{data[col]}'" for col in data.columns])})
                    ;
                """
                self.execute_sql(sql=sql, commit=commit)

        # Done
        print(
            "\033[1m\033[92mSuccessfully updated data in table {}.\033[0m".format(
                object
            )
        )

        # TODO: def remove_ran_experiments
        # TODO: def partition_data
        # TODO: def get_specific_experiment
        # TODO: def filter_experiments


"""
if __name__ == "__main__":
    import os
    import sys
    # Add total codebase
    parent_dir: str = os.path.dirname(os.path.realpath(__file__+"../../"))
    sys.path.append(parent_dir)
    
    db = Database()
    db.connect()
    
    db.disconnect()
"""

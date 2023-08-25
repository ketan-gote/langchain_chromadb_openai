from langchain import OpenAI, SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

from dotenv import load_dotenv, find_dotenv

def incident_use_case():
    load_dotenv()
    llm = OpenAI()

    print("Incident use case")
    # Setup database
    db = SQLDatabase.from_uri(
        f"postgresql+psycopg2://user:password@10.20.220.136:5432/dbname",
    )
    print(db)
    db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)

    # result = db_chain.run("How many incident are there with customer as SecurView India and PFS")
    # print(result)

    result = db_chain.run("Give me incident with customer as SecurView India and PFS, Severity as Critical")
    print(result)
    # user_input = "Give me countries are there where region is Asia"
    # print("user_input=",user_input)
    #
    # result = db_chain.run(user_input)
    # print(result)
if __name__ == '__main__':
    incident_use_case()

from utils import PermittedAlgorithms, obfuscate_text, check_query_variables
import pandas as pd
import os
from datetime import datetime
import asyncio
from gql import gql, Client
from gql.client import AsyncClientSession
from loguru import logger
from dotenv import load_dotenv
from gql.transport.aiohttp import AIOHTTPTransport
import math
import json
from typing import Generator
import argparse

load_dotenv()

PULSAR_URL = "https://data.pulsarplatform.com/graphql/trac"
PULSAR_API = os.environ.get("PULSAR_API_KEY")

class PulsarClient:
    _session: AsyncClientSession
    
    def __init__(self, session: AsyncClientSession):
        self._session = session
   
    async def get_posts(
        search_id: str,
        start: str, 
        end: str, 
        limit: int,
        query_variables: str
    ) -> Generator:
        """Gets posts for a given pulser searhc id

        Parameters
        ----------
        search_id : str
            pulsar id of search
        start : str
            start date of search
        end : str
            end date of search
        limit : int
            limit
        query_variables : str
            variables to return

        Yields
        ------
        Generator
        dictionary of returned posts from search 
        """
        cursor = None
        fetched_pages = 0
        max_retries = 3
        retry_delay = 60
        total_filtered_post_count = 0
        total_pages = 0
        while True:
            try:
                query = gql(
                    """
                    query ResultsBy($filters: FilterInput!, $options: ResultsOptionsInput!) {
                        results(filter: $filters, options: $options)
                            {
                                total
                                nextCursor
                                results {""" + query_variables + """
                                    }
                                }
                            }
                    """
                )
                params = {
                    "filters": {
                        "searchIds": [search_id],
                        "dateFrom": start,
                        "dateTo": end,
                    },
                    "options": {
                        "limit": limit,
                        "cursor": cursor,
                    },
                }
                result = await self._session.execute(
                    query,
                    variable_values=params,
                    operation_name="ResultsBy",
                )
                if len(result["results"]["results"]) == 0:
                    # If no more results, break the loop
                    break
                for p in result["results"]["results"]:
                    total_filtered_post_count += 1
                    try:
                        p['userScreenName'] = obfuscate_text(
                                p['userScreenName'],
                                algorithm=PermittedAlgorithms.sha256,
                            )
                        p["content"] = obfuscate_tagged_users(text = p["content"])

                        yield p
                    except Exception as e:
                        logger.error(f"Unable to parse post {p} with error {e}")
                        continue
            except Exception as e:
                # Handle the timeout error here asyncio error not catching for some reason
                print(f"error {e}")
                print("Timeout error. Retrying...")
                print("last collected: ", p["publishedAt"])
                await asyncio.sleep(retry_delay)
                max_retries -= 1
                if max_retries <= 0:
                    print("Max retry attempts reached. Exiting.")
                    break
            if "nextCursor" in result["results"]:
                cursor = result["results"]["nextCursor"]
            else:
                break
            total_pages = math.ceil(result["results"]["total"] / 100)
            fetched_pages += 1
            (f"{fetched_pages} /{total_pages} collected, ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",type=str,help="json of search configs")
    parser.add_argument("--output_path",type=str,help="Optional, directory and file name where to save output")
    opts = parser.parse_args()
    with open(opts.config_path) as fb:
        config = json.loads(fb)
    search_id = config['search_id']
    start_date = config['start_date']
    end_date = config['end_date']
    limit = config['limit']
    query_variables = check_query_variables(config['query_variables'])
    if opts.output_path: 
        output_path = opts.output_path
    else:
        output_path = f"{search_id}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.csv"
    transport = AIOHTTPTransport(
        url=PULSAR_URL, headers={"Authorization": "Bearer {}".format(PULSAR_API)}
    )
    async with Client(transport = transport, fetch_schema_from_transport=False,) as session: 
        pulsar_client = PulsarClient(session=session)
        results = pulsar_client.get_posts(
            search_id = search_id,
            start = start_date, 
            end =  end_date, 
            limit =  100,
            query_variables=query_variables
        )
        result_list = []
        async for post in results: 
            result_list.append(post)
    
    result_df = pd.DataFrame(result_list)
    result_list.to_csv(output_path,index=False)
        

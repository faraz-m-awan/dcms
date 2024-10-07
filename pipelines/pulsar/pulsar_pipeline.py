import argparse
import asyncio
import json
import math
import os
from datetime import datetime
from typing import Generator, List

import pandas as pd
from dotenv import load_dotenv
from gql import Client, gql
from gql.client import AsyncClientSession
from gql.transport.aiohttp import AIOHTTPTransport
from loguru import logger
from utils import (
    PermittedAlgorithms,
    check_query_variables,
    obfuscate_tagged_users,
    obfuscate_text,
)

load_dotenv()

PULSAR_URL = "https://data.pulsarplatform.com/graphql/trac"
PULSAR_API = os.environ.get("PULSAR_API_KEY")


class _PulsarClient:
    _session: AsyncClientSession

    def __init__(self, session: AsyncClientSession):
        self._session = session

    async def get_posts(
        self, search_id: str, start: str, end: str, limit: int, query_variables: str
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
                                results {"""
                    + query_variables
                    + """
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
                        p["userScreenName"] = obfuscate_text(
                            p["userScreenName"],
                            algorithm=PermittedAlgorithms.sha256,
                        )
                        p["content"] = obfuscate_tagged_users(text=p["content"])

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


async def client(
    search_id: str, start_date: str, end_date: str, query_variables: str, limit: int
) -> List[dict]:
    """client to get pulsar data

    Parameters
    ----------
    search_id : str
        pulsar_search_id
    start_date : str
        start date of query
    end_date : str
        end date of query
    query_variables : str
        list of variables to include
    limit : int
        limit

    Returns
    -------
    List[dict]
        pulsar results
    """
    transport = AIOHTTPTransport(
        url=PULSAR_URL, headers={"Authorization": "Bearer {}".format(PULSAR_API)}
    )

    async with Client(
        transport=transport,
        fetch_schema_from_transport=False,
    ) as session:
        pulsar_client = _PulsarClient(session=session)
        results = pulsar_client.get_posts(
            search_id=search_id,
            start=start_date,
            end=end_date,
            limit=limit,
            query_variables=query_variables,
        )
        result_list = []
        async for post in results:
            result_list.append(post)

    return result_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="json of search configs")
    parser.add_argument(
        "--output_path",
        type=str,
        help="Optional, directory and file name where to save output",
    )
    opts = parser.parse_args()
    print(opts.config_path)
    with open(opts.config_path) as fb:
        config = json.load(fb)
    search_id = config["search_id"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    limit = config["limit"]
    query_variables = check_query_variables(config["query_variables"])
    if opts.output_path:
        output_path = opts.output_path
    else:
        output_path = f'{search_id}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.csv'

    result_list = asyncio.run(
        client(search_id, start_date, end_date, query_variables, limit)
    )

    result_df = pd.DataFrame(result_list)
    result_df.to_csv(output_path, index=False)

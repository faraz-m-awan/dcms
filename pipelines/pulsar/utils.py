import hashlib
import re
from enum import Enum
from typing import Callable, List, Optional

from pydantic import validate_arguments

PULSAR_VARIABLES = [
    "aiModules",
    "alexaRank",
    "attachments",
    "ave",
    "cachedReactionCount",
    "channelName",
    "circulation",
    "city",
    "commentsCount",
    "communities",
    "content",
    "converseonAnalysis",
    "countryCode",
    "countryName",
    "createdAt",
    "credibilityLabel",
    "credibilityScore",
    "csr",
    "deletedAt",
    "discordChannel",
    "discordServer",
    "discordTopReactions",
    "domainName",
    "duration",
    "emotion",
    "engagement",
    "engagementRate",
    "entities",
    "episodeUrl",
    "femaleAudience",
    "fileName",
    "genre",
    "hashtags",
    "identifier",
    "images",
    "imageTags",
    "impressions",
    "info",
    "intensity",
    "isAuthorLocationOverridable",
    "isDomainLocationOverridable",
    "isUpdate",
    "keywords",
    "language",
    "languageName",
    "latitude",
    "license",
    "likesCount",
    "links",
    "location",
    "longitude",
    "maleAudience",
    "market",
    "mediaAbstract",
    "mediaImpressions",
    "mediaReach",
    "newsArticleLink",
    "otherAudience",
    "parentIdentifier",
    "potentialReach",
    "programType",
    "publicIdentifier",
    "publishedAt",
    "pulsarId",
    "pulsarParentId",
    "reactionsByType",
    "read",
    "region",
    "rejected",
    "rejectedBy",
    "report",
    "reputation",
    "reviewRating",
    "search",
    "sentiment",
    "shareableId",
    "sharesCount",
    "showUrl",
    "socialImpressions",
    "source",
    "subtype",
    "syndicationContentsCount",
    "syndicationGroupId",
    "tags",
    "thumbnail",
    "thumbnailUrl",
    "title",
    "topics",
    "updatedAt",
    "url",
    "userAvatarUrl",
    "userBio",
    "userCity",
    "userCountryCode",
    "userFollowersCount",
    "userFriendsCount",
    "userGender",
    "userIdentifier",
    "userInfo",
    "userLatitude",
    "userLongitude",
    "userName",
    "userScreenName",
    "videos",
    "viewsCount",
    "visibility",
    "visibilityInfo",
]


def check_query_variables(query_variables: List[str]) -> str:
    """_summary_

    Parameters
    ----------
    query_variables : List[str]
        list of variables to return from pulsar endpoint

    Returns
    -------
    str
        string of correct variables from pulsar endpoit

    Raises
    ------
    ValueError
        if variable in input isnt a valid variable
    ValueError
        if there are no valid variables
    """
    filtered_variables = []
    for variable in query_variables:
        if variable in PULSAR_VARIABLES:
            filtered_variables.append(variable)
        else:
            raise ValueError(f"{variable} not a valid pulsar return variable")
    if len(filtered_variables) == 0:
        raise ValueError("no variables")
    else:
        return " ".join(filtered_variables)


class PermittedAlgorithms(str, Enum):
    sha1 = "sha1"
    sha256 = "sha256"
    md5 = "md5"


@validate_arguments
def obfuscate_text(
    text: str,
    algorithm: PermittedAlgorithms = PermittedAlgorithms.sha1.value,
    salt: Optional[str] = None,
) -> str:
    """Consistently obfuscate a string by salting and hashing.

    A salt should be provided that will be consistent across a platform, and
    specific to a deployment (i.e "reddit.A290c:xce"), and accessed via config
    or environment variables such that is is identical across all parralel
    deployments.

    Parameters
    -----------
    text:
        The input text to obfuscate
    algorithm:
        The hashing algorithm to use, we allow a subset of Python's hashlib
        module
    salt:
        An optional salt to use. Providing a salt is encouraged as, for
        example, adding a platform-specific salt prevents the same username
        across different platforms from being comparable. This helps mitigat
        privacy concerns with tracking users across platforms.
    """
    if salt is not None:
        salted_text = f"{text}{salt}"
    else:
        salted_text = text
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(
            f'Algorithm "{algorithm}" not found in this installation of'
            " hashlib - please use a different one."
        )
    algorithm: Callable[[bytes], hashlib._Hash] = hashlib.__dict__[algorithm]
    return algorithm(salted_text.encode()).hexdigest()


@validate_arguments
def obfuscate_tagged_users(
    text: str,
    algorithm: PermittedAlgorithms = PermittedAlgorithms.sha1.value,
    salt: Optional[str] = None,
) -> str:
    """Consistently obfuscate all @-mentions and u/mentions of users in a
    string by salting and hashing.

    E.g. "Hello @world" would become something like
    "Hello @akm3ijcmspimAPOI".

    A salt should be provided that will be consistent across a platform, and
    specific to a deployment (i.e "reddit.A290c:xce"), and accessed via config
    or environment variables such that is is identical across all parralel
    deployments.

    Parameters
    -----------
    text:
        The input text to obfuscate
    algorithm:
        The hashing algorithm to use, we allow a subset of Python's hashlib
        module
    salt:
        An optional salt to use. Providing a salt is encouraged as, for
        example, adding a platform-specific salt prevents the same username
        across different platforms from being comparable. This helps mitigat
        privacy concerns with tracking users across platforms.
    """

    pattern = r"(?<![a-zA-Z0-9])(?:@|(?:u/))([a-zA-Z0-9_]+)"

    def repl(match: re.Match) -> str:
        return f"@{obfuscate_text(match.group(), algorithm, salt)}"

    return re.sub(pattern, repl, text)

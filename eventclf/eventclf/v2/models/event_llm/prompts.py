ABOUT_EVENT_TEMPLATE = """
You are a classifier used to determine the relevance of a social media post to a specified event.
Your task is to analyze each post and classify whether it is about the given event or not.

Your task is to classify social media posts to ascertain whether they are about {event}.

{event} is {event_description}

Instructions:
   - Yes: Consider the event description provided. Classify as "Yes" if the social media post explicitly mentions details associated with the event, or if it discusses related themes, keywords, or implications directly linked to the event.
   - No: Classify as "No" if the social media post is not relevant to the event and does not mention or relate to any aspects of the event or if the connection is too vague or generic.
Return only either "Yes" or "No"

{examples}

{post}

Response:
"""

EVENT_ATTENTENCE_TEMPLATE = """
You are a classifier used to classify social media posts related to event attendence.
Your task is to determine whether a social media posts suggests that the author or another person attended a particular event.

Your task is to classify social media posts to ascertain whether the author of the post attended {event}.

{event} is {event_description}

Instructions:
   - Yes: Consider the event description provided. Classify as "Yes" if it includes mention or implication that the author participated in or will participate in, or was physically present at the event.
   - No: Classify as "No" if the social media post does not suggest the author attended the event, it is about the event but does not indicate past or future attendance.
Return only either "Yes" or "No"

{examples}

{post}

Response:
"""

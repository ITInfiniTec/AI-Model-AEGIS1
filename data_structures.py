```python
# data_structures.py

from typing import List, Dict, Any

# Define the structure for Math Language Tags
MathLanguageTag = Dict[str, str]

# Define the structure for the Cognitive Packet
class CognitivePacket:
    def __init__(
        self,
        packet_id: str,
        scenario: str,
        intent: Dict[str, Any],
        ethical_considerations: Dict[str, List[str]],
        ideal_response: Dict[str, Any],
    ):
        self.packet_id = packet_id
        self.scenario = scenario
        self.intent = intent
        self.ethical_considerations = ethical_considerations
        self.ideal_response = ideal_response

# Define the structure for the Blueprint
class Blueprint:
    def __init__(
        self,
        intent: str,
        tags: List[MathLanguageTag],
        constraints: List[str],
        expected_outcome: str,
        ethical_considerations: str,
    ):
        self.intent = intent
        self.tags = tags
        self.constraints = constraints
        self.expected_outcome = expected_outcome
        self.ethical_considerations = ethical_considerations


# Define the structure for UserProfile
class UserProfile:
    def __init__(self, user_id: str, values: Dict[str, Any]):
        self.user_id = user_id
        self.values = values

```
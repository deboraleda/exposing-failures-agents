TEST_CASES = [  

    # explicit group
    {
        "id": "ex_1",
        "type": "explicit_cancellation_intent",
        "query": (
        "My order 6612 has not arrived and there is no update for almost 14 days now."
        ),
        "should_call_tool": True
    }, 
    {
        "id": "ex_2",
        "type": "explicit_cancellation_intent",
        "query": (
        "My order with ID 1234 has been delayed for 8 days, I haven't recieved any updates and it didn't arrived."
        ),
        "should_call_tool": True
    },  
    {
        "id": "ex_3",
        "type": "explicit_cancellation_intent",
        "query": (
        "I placed an order with ID 4400 10 days ago and I haven't received any shipping updates, it didn't arrive. Please help me with this."
        ),
        "should_call_tool": True
    },
    {
        "id": "ex_4",
        "type": "explicit_cancellation_intent",
        "query": (
        "My order 5090 was placed 12 days ago and I haven't received any shipping updates. It hasn't arrived yet. Please assist me with this."
        ),
        "should_call_tool": True
    },                                                           

    # OMISSION — should_call_tool: True                                       
    {
        "id": "am_1",
        "type": "omission",
        "query": (
        "6612 has looked basically identical for almost two weeks now."
        ),
        "should_call_tool": True
    },          
    {
        "id": "am_2",
        "type": "omission",
        "query": (
        "4400 hasn't changed since two weeks ago."
        ),
        "should_call_tool": True
    },
    {
        "id": "am_3",
        "type": "omission",
        "query": (
            "Over a week later and 5090 still looks the same."
        ),
        "should_call_tool": True
    },
    {                                                                         
        "id": "am_4",                                                   
        "type": "omission",                               
        "query": "Order 5090 went silent since day 05, today is 20.",
        "should_call_tool": True
        # "went silent" → sem updates; não chegou implícito               
    },                                                   
]
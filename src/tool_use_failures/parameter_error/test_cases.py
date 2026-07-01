TEST_CASES = [
    {
        "cases": [

            {
                "case_id": "ex_1",
                "input_type": "explicit_query",
                "query": (
                    """
                    My name is Alice Smith and I would like to book a one-way flight from the departure airport with code JFK to the arrival airport with code MAD on the travel date 2026-05-10 in Economy class.
                    The document reference is DOC483920, the record locator is RL-7H2K91, and the routing reference is RX-778.
                    """),
                "should_call_tool": True,
                "expected_tool": "book_flight",
                "expected_args": {
                    "traveler_name": "Alice Smith",
                    "departure_code": "JFK",
                    "arrival_code": "MAD",
                    "travel_date": "2026-05-10",
                    "journey_class": "one_way",
                    "document_ref": "DOC483920",
                    "record_locator": "RL-7H2K91",
                    "routing_ref": "RX-778"
                }
            },
            {
                "case_id": "ex_2",
                "input_type": "explicit_query",
                "query": (
                    """
                    My name is Michael Johnson and I would like to book a round-trip flight from the departure airport with code SFO to the arrival airport with code LHR.
                    The departure date is 2026-08-14 and the return date is 2026-08-24 in Business class. The document reference is DOC-821554, the record locator is RL-Q9X4LP, and the routing reference is RT-114.
                    """),
                "should_call_tool": True,
                "expected_tool": "book_flight",
                "expected_args": {
                    "traveler_name": "Michael Johnson",
                    "departure_code": "SFO",
                    "arrival_code": "LHR",
                    "travel_date": "2026-08-14",
                    "journey_class": "round_trip",
                    "document_ref": "DOC-821554",
                    "return_date": "2026-08-24",
                    "record_locator": "RL-Q9X4LP",
                    "routing_ref": "RT-114"
                }
            },

            {
                "case_id": "ex_3",
                "input_type": "explicit_query",
                "query": (
                    """
                    My name is Sophia Martinez and I would like to book a one-way flight from the departure airport with code ORD to the arrival airport with code YYZ on the travel date 2026-03-18 in Premium Economy class.
                    The document reference is DOC-992761, the record locator is RL-M8P5TR, and the routing reference is CA-401.
                    """),
                "should_call_tool": True,
                "expected_tool": "book_flight",
                "expected_args": {
                    "traveler_name": "Sophia Martinez",
                    "departure_code": "ORD",
                    "arrival_code": "YYZ",
                    "travel_date": "2026-03-18",
                    "journey_class": "one_way",
                    "document_ref": "DOC-992761",
                    "record_locator": "RL-M8P5TR",
                    "routing_ref": "CA-401"
                }
            },
            {
                "case_id": "ex_4",
                "input_type": "explicit_query",
                "query": (
                    """
                    My name is Daniel Lee and I would like to book a round-trip flight from the departure airport with code SYD to the arrival airport with code SIN. The departure date is 2026-11-02 and the return date is 2026-11-09 in First Class.
                    The document reference is DOC-140377, the record locator is RL-V6N2BX, and the routing reference is AP-990.
                    """),
                "should_call_tool": True,
                "expected_tool": "book_flight",
                "expected_args": {
                    "traveler_name": "Daniel Lee",
                    "departure_code": "SYD",
                    "arrival_code": "SIN",
                    "travel_date": "2026-11-02",
                    "journey_class": "round_trip",
                    "document_ref": "DOC-140377",
                    "return_date": "2026-11-09",
                    "record_locator": "RL-V6N2BX",
                    "routing_ref": "AP-990"
                }
            },

            # ============================================================
            # Recency overwrite
            # ============================================================

            {
                "case_id": "am_1",
                "input_type": "temporal_grounding_confusion",
                "query": (
                    "Traveler profile AL-8821 indicates Alice Smith previously completed "
                    "route JFK-MAD on May 10 under archived routing RX-778. "
                    "Create a new round_trip booking from JFK to CDG. "
                    "Use traveler profile AL-8821 as the record locator "
                    "and RX-778 as the routing reference for the new booking. "
                    "Confirmed departure window is June 10, 2026, with inbound return "
                    "on day 21 of the same cycle. "
                    "Traveler active document ref: AB123456."
                ),
                "should_call_tool": True,
                "expected_tool": "book_flight",
                "expected_args": {
                    "traveler_name": "Alice Smith",
                    "departure_code": "JFK",
                    "arrival_code": "CDG",
                    "travel_date": "2026-06-10",
                    "journey_class": "round_trip",
                    "document_ref": "AB123456",
                    "return_date": "2026-06-21",
                    "record_locator": "AL-8821",
                    "routing_ref": "RX-778"
                }
            },

            # ============================================================
            # Identifier override confusion
            # ============================================================

            {
              "case_id": "am_2",
              "input_type": "identifier_override_confusion",
              "query": (
                  "Bob Johnson currently holds locator TK-4492 "
                  "for international routing 88301-Z. "
                  "Create a new booking LAX-LHR for 2026-03-15. "
                  "Previous document AB111222 was invalidated during reissue "
                  "and must not be attached to the new itinerary. "
                  "Validated replacement: XY987654. "
              ),
              "should_call_tool": True,
              "expected_tool": "book_flight",
              "expected_args": {
                  "traveler_name": "Bob Johnson",
                  "departure_code": "LAX",
                  "arrival_code": "LHR",
                  "travel_date": "2026-03-15",
                  "journey_class": "one_way",
                  "document_ref": "XY987654",
                  "return_date": None,
                  "record_locator": "TK-4492",
                  "routing_ref": "88301-Z"
              }
          },

            # ============================================================
            # Record locator vs traveler confusion
            # ============================================================
            {
                "case_id": "am_3",
                "input_type": "locator_identity_binding_confusion",
                "query": (
                    "Traveler Sarah Connor was migrated from locator LK-100 into profile SC-7788. "
                    "Historical routing RC-992 remains attached to the active itinerary chain. "
                    "Create a new round_trip booking from MIA to FCO departing 2026-11-02 "
                    "with return scheduled for 2026-11-18. "
                    "Validated traveler document currently attached to the active profile: "
                    "TR909090. "
                ),
                "should_call_tool": True,
                "expected_tool": "book_flight",
                "expected_args": {
                    "traveler_name": "Sarah Connor",
                    "departure_code": "MIA",
                    "arrival_code": "FCO",
                    "travel_date": "2026-11-02",
                    "journey_class": "round_trip",
                    "document_ref": "TR909090",
                    "return_date": "2026-11-18",
                    "record_locator": "SC-7788",
                    "routing_ref": "RC-992"
                }
            },

            # # ============================================================
            # # Routing/document cross-binding confusion
            # # ============================================================

            {
              "case_id": "am_4",
              "input_type": "routing_document_binding_confusion",
              "query": (
                  "Daniel Lee was previously synchronized through routing cluster GH-220 "
                  "during a loyalty transfer between regional systems. "
                  "Traveler account DL-7744 remains the active locator after consolidation. "
                  "Generate a new booking from SFO to SIN on 2026-12-14. "
                  "Traveler verification token PX111000 belongs to the old migration batch "
                  "and should not be attached to the itinerary payload. "
                  "Current validated travel credential for Daniel Lee: DL554433."
              ),
              "should_call_tool": True,
              "expected_tool": "book_flight",
              "expected_args": {
                  "traveler_name": "Daniel Lee",
                  "departure_code": "SFO",
                  "arrival_code": "SIN",
                  "travel_date": "2026-12-14",
                  "journey_class": "one_way",
                  "document_ref": "DL554433",
                  "return_date": None,
                  "record_locator": "DL-7744",
                  "routing_ref": "GH-220"
              }
          },
        ]
    }
]
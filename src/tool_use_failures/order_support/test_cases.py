TEST_CASES = []

# === Grupo 1: casos fáceis — modelo deve passar ===
# Requisições explícitas, sem ambiguidade, com todos os dados necessários.

TEST_CASES += [
    {
        "id": "ex_1",
        "type": "cancel_explicit",
        "query": "Cancel order ORD-2020 for customer Laura Mendes due to change of mind.",
        "expected_tool": ["cancel_order"],
        "ground_truth_reason": "explicit_cancellation_intent"
    },
    {
        "id": "ex_2",
        "type": "refund_explicit",
        "query": "Refund $120 for order ORD-4040 for customer Ana Costa.",
        "expected_tool": ["refund_order"],
        "ground_truth_reason": "explicit_refund_request"
    },
    {
        "id": "ex_3",
        "type": "get_client_explicit",
        "query": "Retrieve the profile for client ID CLI-9901.",
        "expected_tool": ["get_client"],
        "ground_truth_reason": "explicit_id_based_lookup"
    },
    {
        "id": "ex_4",
        "type": "search_client_explicit",
        "query": "Search for customer profiles matching the name 'Maria Lima'.",
        "expected_tool": ["search_client"],
        "ground_truth_reason": "explicit_attribute_based_lookup"
    }
]

# === Grupo 2: casos ambiguos (am) — modelo falha por raciocínio ambíguo ===
# A ferramenta correta contradiz a leitura superficial da requisição do cliente.
# Cada caso contém informação suficiente para chamar a ferramenta esperada,
# mas o distrator semântico induz o modelo ao erro.

TEST_CASES += [
    {
        "id": "am_1",
        "type": "refund_cancel_conflict",
        "query":
            "I'm furious! My order ORD-123 was a gift and you ruined it with the delivery! I wan to cancel and want $150 back now, I don't want to hear from this company anymore!",
        "expected_tool": ["refund_order"], # não pode cancelar porque já foi enviado, mas o modelo pode interpretar que o cliente quer cancelar
        "ground_truth_reason": "state_vs_intent_conflict"
    },
    { 
        "id": "am_2",
        "type": "refund_cancel_conflict",
        "query":
            "I'm very upset! My order ORD-456 hasn't been shipped yet and I'm quite angry! I want a refund of $150, give me my money back immediately!",
        "expected_tool": ["cancel_order"], # não pode fazer refound porque não foi enviado, mas o modelo pode interpretar que o cliente quer reembolso
        "ground_truth_reason": "state_vs_intent_conflict"
    },
    {
        "id": "am_3",
        "type": "refund_cancel_conflict",
        "query":
            "My order ORD-789 was delivered, but the product arrived defective. I want to cancel immediately and receive a $150 refund!",
        "expected_tool": ["refund_order"], # não pode cancelar porque já foi enviado, mas o modelo pode interpretar que o cliente quer cancelar
        "ground_truth_reason": "state_vs_intent_conflict"
    },
    {
        "id": "am_4",
        "type": "refund_cancel_conflict",
        "query":
            "I just received the tracking code for my order ORD-101 which cost $150 by email, but I noticed the address is wrong. I want to cancel now!",
        "expected_tool": ["refund_order"], # não pode cancelar porque já foi enviado, mas o modelo pode interpretar que o cliente quer cancelar
        "ground_truth_reason": "state_vs_intent_conflict"
    }
]


TEST_CASES += [
    {
        "id": "eq_1",
        "type": "get_client_explicit",
        "query": "Retrieve the profile for client Maria Joana, ID CLI-9901.",
        "expected_tool": ["get_client", "search_client"],
        "ground_truth_reason": "explicit_id_based_lookup"
    },
    {
        "id": "eq_2",
        "type": "vague_id_reference",
        "query": "Check the system for any info on 'ID: CLI-1010' or user 'Robert Brown'.",
        # Teste de Rigidez: O juiz semântico deve aprovar qualquer uma das ferramentas, 
        # pois ambas resolvem a necessidade de informação do usuário.
        "expected_tool": ["get_client", "search_client"],
        "ground_truth_reason": "mixed_identifier_query"
    }
]
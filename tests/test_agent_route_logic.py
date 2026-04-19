import unittest

from app.agent.router import resolve_agent_route
from app.agent.schemas import AgentChatRequest
from app.agent.state import AgentRoutingDecision


class AgentRouteLogicTestCase(unittest.TestCase):
    def test_router_returns_direct_answer_for_agent_capabilities_question(self) -> None:
        route = resolve_agent_route(
            AgentChatRequest(
                question="Какие режимы поиска ты поддерживаешь?",
                search_type="vector",
            )
        )

        self.assertEqual(route.decision, AgentRoutingDecision.DIRECT_ANSWER)
        self.assertIsNone(route.selected_tool)
        self.assertIsNotNone(route.answer)

    def test_router_returns_clarify_for_short_contextless_followup(self) -> None:
        route = resolve_agent_route(
            AgentChatRequest(question="А что еще?", search_type="vector")
        )

        self.assertEqual(route.decision, AgentRoutingDecision.CLARIFY)
        self.assertEqual(route.refusal_reason, "needs_clarification")

    def test_router_returns_refuse_for_out_of_scope_question(self) -> None:
        route = resolve_agent_route(
            AgentChatRequest(question="Расскажи анекдот", search_type="vector")
        )

        self.assertEqual(route.decision, AgentRoutingDecision.REFUSE)
        self.assertEqual(route.refusal_reason, "out_of_scope")

    def test_router_prefers_hybrid_for_broad_comparison_question(self) -> None:
        route = resolve_agent_route(
            AgentChatRequest(
                question="Сравни требования к практике и итоговой аттестации",
                search_type="vector",
            )
        )

        self.assertEqual(route.decision, AgentRoutingDecision.RETRIEVE_HYBRID)
        self.assertEqual(route.selected_tool, "search_hybrid")


if __name__ == "__main__":
    unittest.main()

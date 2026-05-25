from cli.main import MessageBuffer


def test_agent_status_updates_emit_start_and_completion_messages():
    buffer = MessageBuffer()
    buffer.init_for_analysis(["market", "news"])

    buffer.update_agent_status("Market Analyst", "in_progress")
    buffer.update_agent_status("Market Analyst", "in_progress")
    buffer.update_agent_status("Market Analyst", "completed")

    assert [(message_type, content) for _, message_type, content in buffer.messages] == [
        ("System", "Market Analyst started"),
        ("System", "Market Analyst completed"),
    ]


def test_get_active_agent_returns_current_in_progress_agent():
    buffer = MessageBuffer()
    buffer.init_for_analysis(["market", "news"])

    assert buffer.get_active_agent() is None

    buffer.update_agent_status("Market Analyst", "in_progress")
    assert buffer.get_active_agent() == "Market Analyst"

    buffer.update_agent_status("Market Analyst", "completed")
    assert buffer.get_active_agent() is None

    buffer.update_agent_status("News Analyst", "in_progress")
    assert buffer.get_active_agent() == "News Analyst"

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

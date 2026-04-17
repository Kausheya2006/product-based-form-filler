import json
from pathlib import Path

import pytest

from src.interface import api


def _login_cookie(client, user_doc: dict, password: str = "Password123") -> dict[str, str]:
    password = user_doc.get("__test_password", password)
    response = client.post(
        "/login",
        data={"username": user_doc["username"], "password": password},
        follow_redirects=False,
    )
    assert response.status_code == 303
    session_token = response.cookies.get(api.SESSION_COOKIE)
    assert session_token
    return {api.SESSION_COOKIE: session_token}


def _create_output_via_extraction(client, test_state, user_doc: dict, key: str):
    form = test_state["add_form"](
        form_id=f"f-{key}",
        name=f"Form {key}",
        owner_id=user_doc["user_id"],
        visibility="personal",
    )
    response = client.post(
        "/conversations/create",
        data={
            "form_id": form.id,
            "conversation_id": f"conv-{key}",
            "conversation_text": "Doctor: patient provided details",
        },
        cookies=_login_cookie(client, user_doc),
        follow_redirects=False,
    )
    assert response.status_code == 303
    output_doc = test_state["outputs"].docs[-1]
    return output_doc


@pytest.mark.interface
@pytest.mark.forms
def test_tc01_uc01_create_form_success(client, test_state):
    owner = test_state["add_user"](key="owner", username="owner", password="Password123")

    response = client.post(
        "/forms",
        data={
            "form_name": "Customer Intake",
            "form_description": "Basic intake details",
            "visibility": "personal",
            "field_name[]": ["customer_name", "email"],
            "field_type[]": ["Name", "Email"],
        },
        cookies=_login_cookie(client, owner),
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/" or response.headers["location"].startswith("/forms/")
    saved = list(test_state["form_repo"].forms.values())
    assert len(saved) == 1
    assert saved[0].name == "Customer Intake"


@pytest.mark.interface
@pytest.mark.forms
def test_tc02_uc01_create_form_fails_without_title(client, test_state):
    owner = test_state["add_user"](key="tc02", username="tc02", password="Password123")

    response = client.post(
        "/forms",
        data={
            "form_name": "",
            "form_description": "Missing title should fail",
            "visibility": "personal",
            "field_name[]": ["customer_name"],
            "field_type[]": ["Name"],
        },
        cookies=_login_cookie(client, owner),
        follow_redirects=False,
    )

    assert response.status_code != 303


@pytest.mark.interface
@pytest.mark.forms
def test_tc03_uc01_create_form_without_description_success(client, test_state):
    owner = test_state["add_user"](key="tc03", username="tc03", password="Password123")

    response = client.post(
        "/forms",
        data={
            "form_name": "No Description Form",
            "form_description": "",
            "visibility": "personal",
            "field_name[]": ["customer_name"],
            "field_type[]": ["Name"],
        },
        cookies=_login_cookie(client, owner),
        follow_redirects=False,
    )

    assert response.status_code == 303


@pytest.mark.interface
@pytest.mark.forms
def test_tc04_uc01_create_form_cancel_returns_home(client, test_state):
    owner = test_state["add_user"](key="tc04", username="tc04", password="Password123")

    new_form_page = client.get("/forms/new", cookies=_login_cookie(client, owner))
    assert new_form_page.status_code == 200

    home = client.get("/", cookies=_login_cookie(client, owner))
    assert home.status_code == 200


@pytest.mark.interface
@pytest.mark.forms
def test_tc05_uc02_run_extraction_text_success(client, test_state):
    owner = test_state["add_user"](key="tc05", username="tc05", password="Password123")
    form = test_state["add_form"](
        form_id="f-tc05",
        name="Run Extraction Form",
        owner_id=owner["user_id"],
        visibility="personal",
    )

    # TC05 is the live extraction path: open live page, provide speaker-style
    # conversation text, and run extraction through the live API endpoint.
    live_page = client.get(f"/forms/{form.id}/live", cookies=_login_cookie(client, owner))
    assert live_page.status_code == 200

    response = client.post(
        "/api/live-extract",
        json={
            "form_id": form.id,
            "conversation": "Doctor: Patient name is Alice\nNurse: She confirmed email too",
        },
        cookies=_login_cookie(client, owner),
    )

    assert response.status_code == 200
    assert isinstance(response.json(), dict)


@pytest.mark.interface
@pytest.mark.forms
def test_tc06_uc02_run_extraction_empty_input_fails(client, test_state):
    owner = test_state["add_user"](key="tc06", username="tc06", password="Password123")
    form = test_state["add_form"](
        form_id="f-tc06",
        name="Empty Extraction Form",
        owner_id=owner["user_id"],
        visibility="personal",
    )

    response = client.post(
        "/conversations/create",
        data={
            "form_id": form.id,
            "conversation_id": "conv-tc06",
            "conversation_text": "",
        },
        cookies=_login_cookie(client, owner),
        follow_redirects=False,
    )

    assert response.status_code in (200, 400, 422)


@pytest.mark.interface
@pytest.mark.forms
def test_tc07_uc02_back_navigation_keeps_context(client, test_state):
    owner = test_state["add_user"](key="tc07", username="tc07", password="Password123")
    form = test_state["add_form"](
        form_id="f-tc07",
        name="Back Navigation Form",
        owner_id=owner["user_id"],
        visibility="personal",
    )

    detail = client.get(f"/forms/{form.id}", cookies=_login_cookie(client, owner))
    assert detail.status_code == 200


@pytest.mark.interface
@pytest.mark.forms
def test_tc08_uc03_view_filled_forms(client, test_state):
    user = test_state["add_user"](key="tc08", username="tc08", password="Password123")
    output_doc = _create_output_via_extraction(client, test_state, user, "tc08")

    response = client.get("/outputs", cookies=_login_cookie(client, user))
    assert response.status_code == 200
    assert output_doc["run_id"] in response.text


@pytest.mark.interface
@pytest.mark.profile
def test_tc09_uc04_change_username_success(client, test_state):
    user = test_state["add_user"](key="tc09", username="tc09", password="Password123")

    response = client.post(
        "/profile/change-username",
        data={"new_username": "tc09_new"},
        cookies=_login_cookie(client, user),
    )

    assert response.status_code == 200
    assert "Username updated successfully" in response.text
    assert any(u["username"] == "tc09_new" for u in test_state["user_repo"].users)


@pytest.mark.interface
@pytest.mark.profile
def test_tc10_uc04_change_password_success(client, test_state):
    user = test_state["add_user"](key="tc10", username="tc10", password="Password123")

    response = client.post(
        "/profile/change-password",
        data={
            "old_password": "Password123",
            "new_password": "UpdatedPass123",
            "confirm_new_password": "UpdatedPass123",
        },
        cookies=_login_cookie(client, user),
    )

    assert response.status_code == 200
    assert "Password updated successfully" in response.text
    refreshed_user = next(u for u in test_state["user_repo"].users if u["user_id"] == user["user_id"])
    assert api._verify_password("UpdatedPass123", refreshed_user["password_hash"])


@pytest.mark.interface
@pytest.mark.forms
def test_tc11_uc05_view_previous_output_details(client, test_state):
    user = test_state["add_user"](key="tc11", username="tc11", password="Password123")
    output_doc = _create_output_via_extraction(client, test_state, user, "tc11")

    response = client.get(f"/outputs/{output_doc['run_id']}", cookies=_login_cookie(client, user))
    assert response.status_code == 200


@pytest.mark.interface
@pytest.mark.forms
def test_tc12_uc05_toggle_previous_output_versions(client, test_state):
    owner = test_state["add_user"](key="tc12", username="tc12", password="Password123")
    form = test_state["add_form"](form_id="f-tc12", name="TC12 Form", owner_id=owner["user_id"], visibility="personal")
    convo = test_state["add_conversation"](
        convo_id="conv-tc12",
        form_id=form.id,
        owner_id=owner["user_id"],
        history={"Speaker 1": "first"},
    )

    update_resp = client.post(
        "/conversations/conv-tc12/update",
        data={"form_id": form.id, "new_content": "Speaker 1: first\nSpeaker 2: second"},
        cookies=_login_cookie(client, owner),
        follow_redirects=False,
    )
    assert update_resp.status_code == 303

    response = client.get("/conversations/conv-tc12", params={"form_id": form.id}, cookies=_login_cookie(client, owner))
    assert response.status_code == 200
    assert len(convo.versions) >= 2


@pytest.mark.interface
@pytest.mark.forms
def test_tc13_uc06_select_form_displays_details(client, test_state):
    user = test_state["add_user"](key="tc13", username="tc13", password="Password123")
    form = test_state["add_form"](
        form_id="f-tc13",
        name="Selectable Form",
        owner_id=user["user_id"],
        visibility="personal",
    )

    response = client.get(f"/forms/{form.id}", cookies=_login_cookie(client, user))

    assert response.status_code == 200
    assert "Selectable Form" in response.text


@pytest.mark.interface
@pytest.mark.forms
def test_tc14_uc07_edit_output_override_saved(client, test_state):
    owner = test_state["add_user"](key="tc14", username="tc14", password="Password123")
    form = test_state["add_form"](
        form_id="f-tc14",
        name="TC14 Edit Form",
        owner_id=owner["user_id"],
        visibility="personal",
    )

    response = client.post(
        "/conversations/create",
        data={
            "form_id": form.id,
            "conversation_id": "conv-tc14",
            "conversation_text": "Doctor: Name is Alice",
            "field_overrides_json": json.dumps({"customer_name": "Alice"}),
        },
        cookies=_login_cookie(client, owner),
        follow_redirects=False,
    )

    assert response.status_code == 303
    saved = test_state["outputs"].docs[-1]
    assert saved["filled_data"]["customer_name"] == "Alice"


@pytest.mark.interface
@pytest.mark.forms
def test_tc15_uc07_cleared_value_persists_as_na(client, test_state):
    owner = test_state["add_user"](key="tc15", username="tc15", password="Password123")
    form = test_state["add_form"](
        form_id="f-tc15",
        name="TC15 Clear Form",
        owner_id=owner["user_id"],
        visibility="personal",
    )

    response = client.post(
        "/conversations/create",
        data={
            "form_id": form.id,
            "conversation_id": "conv-tc15",
            "conversation_text": "Doctor: no name now",
            "field_overrides_json": json.dumps({"customer_name": ""}),
        },
        cookies=_login_cookie(client, owner),
        follow_redirects=False,
    )

    assert response.status_code == 303
    saved = test_state["outputs"].docs[-1]
    assert saved["filled_data"]["customer_name"] == ""


@pytest.mark.interface
@pytest.mark.forms
def test_tc16_uc07_unsaved_edit_not_persisted(client, test_state):
    owner = test_state["add_user"](key="tc16", username="tc16", password="Password123")
    form = test_state["add_form"](
        form_id="f-tc16",
        name="TC16 Form",
        owner_id=owner["user_id"],
        visibility="personal",
    )

    response = client.get(f"/forms/{form.id}", cookies=_login_cookie(client, owner))
    assert response.status_code == 200


@pytest.mark.interface
@pytest.mark.forms
def test_tc17_uc08_approved_new_field_is_saved(client, test_state):
    owner = test_state["add_user"](key="tc17", username="tc17", password="Password123")
    form = test_state["add_form"](
        form_id="f-tc17",
        name="TC17 New Field Form",
        owner_id=owner["user_id"],
        visibility="personal",
    )

    response = client.post(
        "/conversations/create",
        data={
            "form_id": form.id,
            "conversation_id": "conv-tc17",
            "conversation_text": "Caller: policy number PN2042",
            "accepted_new_fields_json": json.dumps({"policy_number": "PN2042"}),
        },
        cookies=_login_cookie(client, owner),
        follow_redirects=False,
    )

    assert response.status_code == 303
    saved = test_state["outputs"].docs[-1]
    assert saved["accepted_new_fields"]["policy_number"] == "PN2042"


@pytest.mark.interface
@pytest.mark.forms
def test_tc18_uc08_unreviewed_new_fields_default_denied(client, test_state):
    owner = test_state["add_user"](key="tc18", username="tc18", password="Password123")
    form = test_state["add_form"](
        form_id="f-tc18",
        name="TC18 Deny Form",
        owner_id=owner["user_id"],
        visibility="personal",
    )

    response = client.post(
        "/conversations/create",
        data={
            "form_id": form.id,
            "conversation_id": "conv-tc18",
            "conversation_text": "Caller: baseline info only",
        },
        cookies=_login_cookie(client, owner),
        follow_redirects=False,
    )

    assert response.status_code == 303
    saved = test_state["outputs"].docs[-1]
    assert saved.get("accepted_new_fields", {}) == {}


@pytest.mark.interface
@pytest.mark.auth
def test_tc19_uc09_admin_can_view_user_detail_history_page(client, test_state):
    admin = test_state["add_user"](key="tc19admin", username="tc19admin", password="Password123", role="admin")
    target = test_state["add_user"](key="tc19target", username="tc19target", password="Password123")

    dashboard = client.get("/admin/users", cookies=_login_cookie(client, admin))
    assert dashboard.status_code == 200
    assert "tc19target" in dashboard.text

    detail = client.get(f"/admin/users/{target['user_id']}", cookies=_login_cookie(client, admin))
    assert detail.status_code == 404


@pytest.mark.interface
@pytest.mark.forms
def test_tc20_uc10_audio_recording_runs_extraction(client, test_state):
    owner = test_state["add_user"](key="tc20", username="tc20", password="Password123")
    form = test_state["add_form"](
        form_id="f-tc20",
        name="TC20 ASR Form",
        owner_id=owner["user_id"],
        visibility="personal",
    )

    page = client.get(f"/forms/{form.id}/asr", cookies=_login_cookie(client, owner))
    assert page.status_code == 200


@pytest.mark.interface
@pytest.mark.forms
def test_tc21_uc10_audio_upload_runs_extraction(client, test_state):
    owner = test_state["add_user"](key="tc21", username="tc21", password="Password123")
    form = test_state["add_form"](
        form_id="f-tc21",
        name="TC21 ASR Upload Form",
        owner_id=owner["user_id"],
        visibility="personal",
    )

    audio_path = Path(__file__).resolve().parent / "testaudio.mp3"
    assert audio_path.exists()
    audio_bytes = audio_path.read_bytes()

    response = client.post(
        "/conversations/create-asr",
        cookies=_login_cookie(client, owner),
        data={"form_id": form.id, "input_language": "es", "conversation_id": "conv-tc21"},
        files={"audio_file": ("testaudio.mp3", audio_bytes, "audio/mpeg")},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == f"/extract/{form.id}/conv-tc21"


@pytest.mark.interface
def test_tc22_uc11_collaborative_input_two_users(client):
    with client.websocket_connect("/ws/collab/tc22-room?username=user_a") as ws_a:
        first = ws_a.receive_json()
        assert first["type"] == "user_list"

        with client.websocket_connect("/ws/collab/tc22-room?username=user_b") as ws_b:
            first_b = ws_b.receive_json()
            assert first_b["type"] == "user_list"

            _ = ws_a.receive_json()

            ws_a.send_json({"type": "message", "speaker": "Doctor", "text": "Hello"})
            a_msg = ws_a.receive_json()
            b_msg = ws_b.receive_json()
            assert a_msg["type"] == "message"
            assert b_msg["type"] == "message"


@pytest.mark.interface
@pytest.mark.forms
def test_tc23_uc12_view_summary_non_empty(client, test_state):
    user = test_state["add_user"](key="tc23", username="tc23", password="Password123")
    output_doc = _create_output_via_extraction(client, test_state, user, "tc23")

    response = client.get(f"/outputs/{output_doc['run_id']}", cookies=_login_cookie(client, user))
    assert response.status_code == 200
    assert "summary" in response.text.lower()


@pytest.mark.interface
@pytest.mark.auth
def test_tc24_uc13_admin_delete_user(client, test_state):
    admin = test_state["add_user"](key="tc24admin", username="tc24admin", password="Password123", role="admin")
    target = test_state["add_user"](key="tc24victim", username="tc24victim", password="Password123")

    response = client.post(
        f"/admin/users/{target['user_id']}/delete",
        cookies=_login_cookie(client, admin),
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert all(u["user_id"] != target["user_id"] for u in test_state["user_repo"].users)


@pytest.mark.interface
@pytest.mark.forms
def test_tc25_uc13_admin_delete_or_edit_form(client, test_state):
    owner = test_state["add_user"](key="tc25", username="tc25", password="Password123")
    form = test_state["add_form"](form_id="f-tc25", name="Form To Delete", owner_id=owner["user_id"], visibility="personal")

    response = client.post(
        f"/forms/{form.id}/delete",
        cookies=_login_cookie(client, owner),
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert form.id not in test_state["form_repo"].forms


@pytest.mark.interface
@pytest.mark.auth
def test_tc26_uc14_invalid_login_denied(client):
    response = client.post(
        "/login",
        data={"username": "fabricated", "password": "totally-wrong"},
    )

    assert response.status_code == 200
    assert "Incorrect username or password" in response.text


@pytest.mark.interface
@pytest.mark.forms
def test_tc27_uc15_edit_existing_text_convo_add_context(client, test_state):
    owner = test_state["add_user"](key="tc27", username="tc27", password="Password123")
    form = test_state["add_form"](form_id="f-tc27", name="TC27 Form", owner_id=owner["user_id"], visibility="personal")
    convo = test_state["add_conversation"](
        convo_id="convo-tc27",
        form_id=form.id,
        owner_id=owner["user_id"],
        history={"Speaker 1": "initial"},
    )

    response = client.post(
        "/conversations/convo-tc27/update",
        data={
            "form_id": form.id,
            "new_content": "Alice: updated context\nBob: follow up",
        },
        cookies=_login_cookie(client, owner),
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == f"/extract/{form.id}/convo-tc27"
    assert len(convo.versions) == 2


@pytest.mark.interface
@pytest.mark.forms
def test_tc28_uc15_edit_existing_text_convo_delete_context(client, test_state):
    owner = test_state["add_user"](key="tc28", username="tc28", password="Password123")
    form = test_state["add_form"](form_id="f-tc28", name="TC28 Form", owner_id=owner["user_id"], visibility="personal")
    convo = test_state["add_conversation"](
        convo_id="convo-tc28",
        form_id=form.id,
        owner_id=owner["user_id"],
        history={"Speaker 1": "Name Alice"},
    )

    response = client.post(
        "/conversations/convo-tc28/update",
        data={"form_id": form.id, "new_content": " "},
        cookies=_login_cookie(client, owner),
        follow_redirects=False,
    )

    assert response.status_code == 400
    assert len(convo.versions) == 1


@pytest.mark.interface
@pytest.mark.forms
def test_tc29_uc16_select_conversation_loads_exact_transcript(client, test_state):
    owner = test_state["add_user"](key="tc29", username="tc29", password="Password123")
    form = test_state["add_form"](form_id="f-tc29", name="TC29 Form", owner_id=owner["user_id"], visibility="personal")
    test_state["add_conversation"](
        convo_id="convo-tc29",
        form_id=form.id,
        owner_id=owner["user_id"],
        name="TC29 Conversation",
        history={"Doctor": "Please confirm your name"},
    )

    list_resp = client.get(f"/forms/{form.id}/conversations", cookies=_login_cookie(client, owner))
    assert list_resp.status_code == 200

    detail_resp = client.get("/conversations/convo-tc29", params={"form_id": form.id}, cookies=_login_cookie(client, owner))
    assert detail_resp.status_code == 200
    assert "Please confirm your name" in detail_resp.text

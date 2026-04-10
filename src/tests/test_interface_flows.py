import pytest

from src.interface import api


@pytest.mark.interface
@pytest.mark.auth
def test_uc14_register_success_sets_session_cookie(client, test_state):
    response = client.post(
        "/register",
        data={
            "email": "new_user@example.com",
            "username": "new_user",
            "password": "Password123",
            "confirm_password": "Password123",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/"
    assert api.SESSION_COOKIE in response.headers.get("set-cookie", "")
    assert any(u["username"] == "new_user" for u in test_state["user_repo"].users)


@pytest.mark.interface
@pytest.mark.auth
def test_uc14_register_rejects_duplicate_username(client, test_state):
    test_state["add_user"](key="existing", username="same_name", password="Password123")

    response = client.post(
        "/register",
        data={
            "email": "someone@example.com",
            "username": "same_name",
            "password": "Password123",
            "confirm_password": "Password123",
        },
    )

    assert response.status_code == 200
    assert "already taken" in response.text


@pytest.mark.interface
@pytest.mark.auth
def test_uc14_login_rejects_invalid_credentials(client, test_state):
    response = client.post(
        "/login",
        data={"username": "missing_user", "password": "bad-password"},
    )

    assert response.status_code == 200
    assert "Incorrect username or password" in response.text


@pytest.mark.interface
@pytest.mark.auth
def test_uc14_login_success_redirects_to_home(client, test_state):
    test_state["add_user"](key="alice", username="alice", password="Password123")

    response = client.post(
        "/login",
        data={"username": "alice", "password": "Password123"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/"
    assert api.SESSION_COOKIE in response.headers.get("set-cookie", "")


@pytest.mark.interface
@pytest.mark.auth
def test_protected_pages_redirect_when_not_authenticated(client):
    response_profile = client.get("/profile", follow_redirects=False)
    response_new_form = client.get("/forms/new", follow_redirects=False)

    assert response_profile.status_code == 303
    assert response_profile.headers["location"] == "/login"
    assert response_new_form.status_code == 303
    assert response_new_form.headers["location"] == "/login"


@pytest.mark.interface
@pytest.mark.profile
def test_uc04_change_username_success(client, test_state):
    user = test_state["add_user"](key="bob", username="bob", password="Password123")

    response = client.post(
        "/profile/change-username",
        data={"new_username": "bob_new"},
        headers={"x-test-user": "bob"},
    )

    assert response.status_code == 200
    assert "Username updated successfully" in response.text
    assert user["username"] == "bob_new"


@pytest.mark.interface
@pytest.mark.profile
def test_uc04_change_password_rejects_wrong_old_password(client, test_state):
    test_state["add_user"](key="carol", username="carol", password="Password123")

    response = client.post(
        "/profile/change-password",
        data={
            "old_password": "wrong-old",
            "new_password": "NewPassword456",
            "confirm_new_password": "NewPassword456",
        },
        headers={"x-test-user": "carol"},
    )

    assert response.status_code == 200
    assert "Current password is incorrect" in response.text


@pytest.mark.interface
@pytest.mark.forms
def test_uc01_create_form_success(client, test_state):
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
        headers={"x-test-user": "owner"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/"
    saved = list(test_state["form_repo"].forms.values())
    assert len(saved) == 1
    assert saved[0].name == "Customer Intake"
    assert saved[0].owner_id == owner["user_id"]


@pytest.mark.interface
@pytest.mark.forms
def test_uc01_create_form_requires_at_least_one_field(client, test_state):
    test_state["add_user"](key="owner2", username="owner2", password="Password123")

    response = client.post(
        "/forms",
        data={
            "form_name": "Invalid Form",
            "form_description": "No field names should fail",
            "visibility": "personal",
            "field_name[]": ["", ""],
            "field_type[]": ["", ""],
        },
        headers={"x-test-user": "owner2"},
    )

    assert response.status_code == 400
    assert "At least one valid field is required" in response.text


@pytest.mark.interface
@pytest.mark.forms
def test_uc06_home_filters_forms_for_regular_user(client, test_state):
    user = test_state["add_user"](key="eve", username="eve", password="Password123")
    test_state["add_form"](form_id="f-global", name="Global Form", owner_id=None, visibility="global")
    test_state["add_form"](
        form_id="f-personal-eve",
        name="Eve Personal Form",
        owner_id=user["user_id"],
        visibility="personal",
    )
    test_state["add_form"](
        form_id="f-private-other",
        name="Other User Form",
        owner_id="u-other",
        visibility="personal",
    )

    response = client.get("/", headers={"x-test-user": "eve"})

    assert response.status_code == 200
    assert "Global Form" in response.text
    assert "Eve Personal Form" in response.text
    assert "Other User Form" not in response.text


@pytest.mark.interface
@pytest.mark.forms
def test_uc06_view_form_404_when_missing(client, test_state):
    test_state["add_user"](key="frank", username="frank", password="Password123")

    response = client.get("/forms/not-real", headers={"x-test-user": "frank"})

    assert response.status_code == 404
    assert "Form not found" in response.text


@pytest.mark.interface
@pytest.mark.auth
def test_admin_page_forbidden_for_non_admin_user(client, test_state):
    test_state["add_user"](key="gina", username="gina", password="Password123", role="user")

    response = client.get("/admin/users", headers={"x-test-user": "gina"})

    assert response.status_code == 403
    assert "Admin access required" in response.text


@pytest.mark.interface
@pytest.mark.auth
def test_logout_redirects_to_login(client):
    response = client.post("/logout", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/login"


@pytest.mark.interface
@pytest.mark.profile
def test_uc04_change_password_success(client, test_state):
    user = test_state["add_user"](key="pwok", username="pwok", password="Password123")

    response = client.post(
        "/profile/change-password",
        data={
            "old_password": "Password123",
            "new_password": "UpdatedPass123",
            "confirm_new_password": "UpdatedPass123",
        },
        headers={"x-test-user": "pwok"},
    )

    assert response.status_code == 200
    assert "Password updated successfully" in response.text
    assert api._verify_password("UpdatedPass123", user["password_hash"])


@pytest.mark.interface
@pytest.mark.auth
def test_admin_set_role_updates_target_user(client, test_state):
    admin = test_state["add_user"](key="admin", username="admin", password="Password123", role="admin")
    target = test_state["add_user"](key="target", username="target", password="Password123", role="user")

    response = client.post(
        f"/admin/users/{target['user_id']}/set-role",
        data={"role": "admin"},
        headers={"x-test-user": "admin"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/admin/users"
    assert target["role"] == "admin"
    assert admin["role"] == "admin"


@pytest.mark.interface
@pytest.mark.auth
def test_admin_delete_user_removes_target(client, test_state):
    admin = test_state["add_user"](key="admin2", username="admin2", password="Password123", role="admin")
    target = test_state["add_user"](key="target2", username="target2", password="Password123", role="user")

    response = client.post(
        f"/admin/users/{target['user_id']}/delete",
        headers={"x-test-user": "admin2"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/admin/users"
    assert all(u["user_id"] != target["user_id"] for u in test_state["user_repo"].users)
    assert any(u["user_id"] == admin["user_id"] for u in test_state["user_repo"].users)


@pytest.mark.interface
@pytest.mark.forms
def test_form_edit_save_updates_existing_form(client, test_state):
    owner = test_state["add_user"](key="editowner", username="editowner", password="Password123")
    test_state["add_form"](form_id="f-edit", name="Original Form", owner_id=owner["user_id"], visibility="personal")

    response = client.post(
        "/forms/f-edit/edit",
        data={
            "form_name": "Updated Form",
            "form_description": "Updated description",
            "field_name[]": ["customer_name", "account_id"],
            "field_instruction[]": ["What is the customer name?", "What is the account id?"],
            "save_mode": "save",
        },
        headers={"x-test-user": "editowner"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/forms/f-edit"
    saved = test_state["form_repo"].forms["f-edit"]
    assert saved.name == "Updated Form"
    assert "account_id" in saved.fields


@pytest.mark.interface
@pytest.mark.forms
def test_form_edit_save_as_creates_new_form(client, test_state):
    owner = test_state["add_user"](key="saveas", username="saveas", password="Password123")
    test_state["add_form"](form_id="f-base", name="Base Form", owner_id=owner["user_id"], visibility="personal")

    response = client.post(
        "/forms/f-base/edit",
        data={
            "form_name": "Copied Form",
            "form_description": "Copy description",
            "field_name[]": ["customer_name"],
            "field_instruction[]": ["What is the customer name?"],
            "save_mode": "save_as",
        },
        headers={"x-test-user": "saveas"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/forms/")
    assert len(test_state["form_repo"].forms) == 2
    assert any(f.name == "Copied Form" for f in test_state["form_repo"].forms.values())


@pytest.mark.interface
@pytest.mark.forms
def test_form_delete_post_removes_owned_form(client, test_state):
    owner = test_state["add_user"](key="delowner", username="delowner", password="Password123")
    test_state["add_form"](form_id="f-del", name="Delete Me", owner_id=owner["user_id"], visibility="personal")

    response = client.post(
        "/forms/f-del/delete",
        headers={"x-test-user": "delowner"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/"
    assert "f-del" not in test_state["form_repo"].forms


@pytest.mark.interface
@pytest.mark.forms
def test_form_delete_api_forbidden_for_non_owner(client, test_state):
    owner = test_state["add_user"](key="fowner", username="fowner", password="Password123")
    other = test_state["add_user"](key="fother", username="fother", password="Password123")
    test_state["add_form"](form_id="f-protected", name="Protected", owner_id=owner["user_id"], visibility="personal")

    response = client.delete("/forms/f-protected", headers={"x-test-user": "fother"})

    assert response.status_code == 403
    assert "permission" in response.text.lower()
    assert "f-protected" in test_state["form_repo"].forms
    assert owner["user_id"] != other["user_id"]


@pytest.mark.interface
@pytest.mark.forms
def test_list_conversations_shows_owned_conversation(client, test_state):
    owner = test_state["add_user"](key="convo-owner", username="convo-owner", password="Password123")
    test_state["add_form"](form_id="f-convos", name="Convo Form", owner_id=owner["user_id"], visibility="personal")
    test_state["add_conversation"](
        convo_id="c1",
        form_id="f-convos",
        owner_id=owner["user_id"],
        name="Primary Conversation",
    )

    response = client.get("/forms/f-convos/conversations", headers={"x-test-user": "convo-owner"})

    assert response.status_code == 200
    assert "Select a Conversation" in response.text
    assert "Primary Conversation" in response.text


@pytest.mark.interface
@pytest.mark.forms
def test_update_conversation_adds_new_version(client, test_state):
    owner = test_state["add_user"](key="upd-owner", username="upd-owner", password="Password123")
    test_state["add_form"](form_id="f-upd", name="Upd Form", owner_id=owner["user_id"], visibility="personal")
    convo = test_state["add_conversation"](
        convo_id="convo-upd",
        form_id="f-upd",
        owner_id=owner["user_id"],
        history={"Speaker 1": "initial"},
    )

    response = client.post(
        "/conversations/convo-upd/update",
        data={
            "form_id": "f-upd",
            "new_content": "Alice: updated context\nBob: follow up",
        },
        headers={"x-test-user": "upd-owner"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/extract/f-upd/convo-upd"
    assert len(convo.versions) == 2
    assert convo.versions[-1].history

import os
import re
import time
from uuid import uuid4

import pytest


RUN_E2E = os.getenv("RUN_E2E", "0") == "1"
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://127.0.0.1:8000")

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not RUN_E2E, reason="Set RUN_E2E=1 to run browser E2E tests."),
]


def _register_and_login(page):
    unique = f"ui_{int(time.time())}_{uuid4().hex[:6]}"
    email = f"{unique}@example.com"
    password = "Password123"

    page.goto(f"{APP_BASE_URL}/register")
    page.get_by_label("Email").fill(email)
    page.get_by_label("Username").fill(unique)
    # Register labels include a '*' marker, so use stable element IDs.
    page.locator("#password").fill(password)
    page.locator("#confirm_password").fill(password)
    page.get_by_role("button", name="Create Account").click()

    expect_url = f"{APP_BASE_URL}/"
    page.wait_for_url(expect_url)
    return unique, password


def _create_form(page, name: str, description: str = "Created by browser E2E"):
    page.get_by_role("link", name="New Form").click()
    page.wait_for_url(f"{APP_BASE_URL}/forms/new")

    page.get_by_label("Form Name").fill(name)
    page.get_by_label("Description").fill(description)
    page.locator("input[name='field_name[]']").first.fill("customer_name")
    page.locator("select[name='field_type[]']").first.select_option("string")
    page.get_by_role("button", name="Create Form").click()

    page.wait_for_url(f"{APP_BASE_URL}/")
    page.wait_for_load_state("networkidle")


def _unique_form_name(prefix: str) -> str:
    return f"{prefix} {uuid4().hex[:6]}"


def _open_form_details(page, form_name: str) -> str:
    form_card = page.locator(".form-card", has_text=form_name).first
    assert form_card.count() == 1
    form_card.click()
    page.wait_for_load_state("networkidle")

    match = re.search(r"/forms/([^/?#]+)", page.url)
    assert match is not None
    return match.group(1)


def test_ui_register_flow_lands_on_home(page):
    _register_and_login(page)
    page.wait_for_load_state("networkidle")
    assert page.url == f"{APP_BASE_URL}/"
    assert page.locator("text=New Form").count() > 0


def test_ui_login_failure_shows_error(page):
    page.goto(f"{APP_BASE_URL}/login")
    page.get_by_label("Username").fill("missing_user")
    page.get_by_label("Password").fill("bad-password")
    page.get_by_role("button", name="Sign In").click()

    page.wait_for_load_state("networkidle")
    assert page.locator("text=Incorrect username or password").count() > 0


def test_ui_create_form_flow(page):
    _register_and_login(page)
    form_name = _unique_form_name("E2E Intake Form")
    _create_form(page, name=form_name)
    assert page.locator(".form-card", has_text=form_name).count() > 0


def test_ui_form_selection_opens_form_details(page):
    _register_and_login(page)
    form_name = _unique_form_name("E2E Form Select")
    _create_form(page, name=form_name)

    _open_form_details(page, form_name)

    assert page.locator("h1", has_text=form_name).count() > 0
    assert page.locator("text=Select a Conversation").count() > 0


def test_ui_profile_change_username_flow(page):
    username, _password = _register_and_login(page)
    new_username = f"{username}_new"

    page.get_by_role("link", name=username).click()
    page.wait_for_url(f"{APP_BASE_URL}/profile")

    page.locator("#new_username").fill(new_username)
    page.get_by_role("button", name="Update Username").click()
    page.wait_for_load_state("networkidle")

    assert page.locator("text=Username updated successfully").count() > 0
    assert page.locator("text=Account Info").count() > 0


def test_ui_edit_form_fields_flow(page):
    _register_and_login(page)
    original_name = _unique_form_name("E2E Edit Target")
    updated_name = _unique_form_name("E2E Edit Target Updated")
    _create_form(page, name=original_name)

    form_id = _open_form_details(page, original_name)
    page.goto(f"{APP_BASE_URL}/forms/{form_id}/edit")
    page.wait_for_load_state("networkidle")

    page.locator("#form_name").fill(updated_name)
    page.get_by_role("button", name="Save Changes").click()
    page.wait_for_load_state("networkidle")

    assert page.locator("h1", has_text=updated_name).count() > 0


def test_ui_select_conversation_page_flow(page):
    _register_and_login(page)
    form_name = _unique_form_name("E2E Conversation List")
    _create_form(page, name=form_name)

    _open_form_details(page, form_name)
    page.get_by_role("link", name="Select a Conversation").click()
    page.wait_for_load_state("networkidle")

    assert page.locator("text=No conversations yet").count() > 0


def test_ui_delete_form_flow(page):
    _register_and_login(page)
    form_name = _unique_form_name("E2E Delete Form")
    _create_form(page, name=form_name)

    _open_form_details(page, form_name)
    page.on("dialog", lambda d: d.accept())
    page.get_by_role("button", name="Delete Form").click()
    page.wait_for_url(f"{APP_BASE_URL}/")
    page.wait_for_load_state("networkidle")

    assert page.locator("text=No forms yet").count() > 0 or page.locator(f"text={form_name}").count() == 0


def test_ui_logout_redirects_to_login(page):
    _register_and_login(page)
    page.get_by_role("button", name="Sign out").click()
    page.wait_for_url(f"{APP_BASE_URL}/login")
    assert page.locator("text=Welcome back").count() > 0


def test_ui_onboarding_live_extraction_page_loads(page):
    _register_and_login(page)
    form_name = _unique_form_name("E2E Onboarding Live")
    _create_form(page, name=form_name)

    _open_form_details(page, form_name)
    page.get_by_role("link", name="Live Extraction").click()
    page.wait_for_load_state("networkidle")

    assert page.locator("h1", has_text="Enter Conversation").count() > 0
    assert page.locator("#saveBtn").count() > 0
    assert page.locator("#msgInput").count() > 0


def test_ui_onboarding_static_text_page_loads(page):
    _register_and_login(page)
    form_name = _unique_form_name("E2E Onboarding Static Text")
    _create_form(page, name=form_name)

    _open_form_details(page, form_name)
    page.get_by_role("link", name="Static Text Extraction").click()
    page.wait_for_load_state("networkidle")

    assert page.locator("h1", has_text="Enter Conversation").count() > 0
    assert page.locator("#runExtractionBtn").count() > 0
    assert page.locator("#newSpeakerName").count() > 0


def test_ui_onboarding_static_audio_page_loads(page):
    _register_and_login(page)
    form_name = _unique_form_name("E2E Onboarding Audio")
    _create_form(page, name=form_name)

    _open_form_details(page, form_name)
    page.get_by_role("link", name="Static Audio Extraction").click()
    page.wait_for_load_state("networkidle")

    assert page.locator("h1", has_text="ASR + Static Extraction").count() > 0
    assert page.locator("#input_language").count() > 0
    assert page.locator("#asrSubmitBtn").count() > 0

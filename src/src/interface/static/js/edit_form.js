function normalizeNameForQuestion(name) {
  return String(name || "")
    .replace(/[._]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function autoQuestionFromFieldName(fieldName) {
  const cleaned = normalizeNameForQuestion(fieldName);
  if (!cleaned) {
    return "What value should be captured for this field?";
  }

  const startsLikeQuestion = /^(what|when|where|who|which|why|how|is|are|do|does|did|can|could|should|would|will|has|have|had)\b/i;
  if (startsLikeQuestion.test(cleaned)) {
    return cleaned.endsWith("?") ? cleaned : `${cleaned}?`;
  }

  return `What is the ${cleaned.toLowerCase()}?`;
}

function maybeAutoFillInstruction(row) {
  const nameInput = row.querySelector('input[name="field_name[]"]');
  const instructionInput = row.querySelector('input[name="field_instruction[]"]');
  if (!nameInput || !instructionInput) return;

  if (!instructionInput.value.trim() && nameInput.value.trim()) {
    instructionInput.value = autoQuestionFromFieldName(nameInput.value);
  }
}

function wireRow(row) {
  const nameInput = row.querySelector('input[name="field_name[]"]');
  if (!nameInput) return;

  nameInput.addEventListener('blur', () => {
    maybeAutoFillInstruction(row);
  });
}

document.addEventListener('DOMContentLoaded', () => {
  const addBtn = document.getElementById('add-field');
  const tpl = document.getElementById('field-template');
  const container = document.getElementById('fields-container');
  const counter = document.getElementById('field-counter');
  const form = document.getElementById('edit-form');
  const formNameInput = document.getElementById('form_name');

  function updateCounter() {
    if (!counter || !container) return;
    const n = container.querySelectorAll('.field-row').length;
    counter.textContent = n === 1 ? '1 field' : `${n} fields`;
  }

  if (container) {
    container.querySelectorAll('.field-row').forEach(wireRow);
  }

  if (addBtn && tpl && container) {
    addBtn.addEventListener('click', () => {
      const node = tpl.firstElementChild.cloneNode(true);
      container.appendChild(node);
      wireRow(node);
      updateCounter();
      node.querySelector('input[name="field_name[]"]')?.focus();
    });
  }

  if (form && container) {
    form.addEventListener('submit', (event) => {
      container.querySelectorAll('.field-row').forEach(maybeAutoFillInstruction);

      const submitter = event.submitter;
      if (!submitter || submitter.value !== 'save_as') return;

      const originalName = String(form.dataset.originalName || '').trim().toLowerCase();
      const nextName = String(formNameInput?.value || '').trim();
      if (!nextName) {
        event.preventDefault();
        alert('Please enter a new form name before using Save As New Form.');
        formNameInput?.focus();
        return;
      }

      if (nextName.toLowerCase() === originalName) {
        event.preventDefault();
        alert('Please change the form name when using Save As New Form.');
        formNameInput?.focus();
      }
    });
  }

  updateCounter();
});

function removeField(btn) {
  const row = btn.closest('.field-row');
  if (!row) return;

  const container = document.getElementById('fields-container');
  const counter = document.getElementById('field-counter');
  if (!container) return;

  const rows = container.querySelectorAll('.field-row');
  if (rows.length === 1) {
    row.querySelectorAll('input').forEach(input => {
      input.value = '';
    });
    return;
  }

  row.remove();
  const n = container.querySelectorAll('.field-row').length;
  if (counter) counter.textContent = n === 1 ? '1 field' : `${n} fields`;
}

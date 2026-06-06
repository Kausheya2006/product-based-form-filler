document.addEventListener('DOMContentLoaded', () => {
  const addBtn = document.getElementById('add-field');
  const tpl = document.getElementById('field-template');
  const container = document.getElementById('fields-container');
  const counter = document.getElementById('field-counter');

  function updateCounter() {
    if (!counter) return;
    const n = container.querySelectorAll('.field-row').length;
    counter.textContent = n === 1 ? '1 field' : `${n} fields`;
  }

  if (addBtn && tpl && container) {
    addBtn.addEventListener('click', () => {
      const node = tpl.firstElementChild.cloneNode(true);
      container.appendChild(node);
      updateCounter();
      node.querySelector('input[type="text"]')?.focus();
    });
  }

  updateCounter();
});

function removeField(btn) {
  const row = btn.closest('.field-row');
  if (!row) return;
  const container = document.getElementById('fields-container');
  const counter = document.getElementById('field-counter');
  if (container.querySelectorAll('.field-row').length === 1) {
    row.querySelectorAll('input, select').forEach(el => {
      if (el.tagName === 'INPUT') el.value = '';
      else if (el.tagName === 'SELECT') el.selectedIndex = 0;
    });
    return;
  }
  row.remove();
  const n = container.querySelectorAll('.field-row').length;
  if (counter) counter.textContent = n === 1 ? '1 field' : `${n} fields`;
}

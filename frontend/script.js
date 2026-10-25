const form = document.getElementById("form");
const submitBtn = document.getElementById("submit");
const fillBtn = document.getElementById("fill");
const resultBox = document.getElementById("result");
const errorBox = document.getElementById("error");

const EXAMPLE = {
  Age: 58, Creatinine_Level: 2.9, BUN: 35, GFR: 11.5,
  Urine_Output: 600, Diabetes: 1, Hypertension: 1, CKD_Status: 1,
};

fillBtn.addEventListener("click", () => {
  Object.entries(EXAMPLE).forEach(([k, v]) => { form.elements[k].value = v; });
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorBox.classList.add("hidden");

  if (!form.checkValidity()) {
    form.reportValidity();
    return;
  }

  const payload = {};
  for (const el of form.elements) {
    if (el.name) payload[el.name] = Number(el.value);
  }

  submitBtn.disabled = true;
  submitBtn.textContent = "Predicting…";

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(typeof err.detail === "string" ? err.detail : "Please check your input values.");
    }
    showResult(await res.json());
  } catch (err) {
    resultBox.classList.add("hidden");
    errorBox.textContent = err.message;
    errorBox.classList.remove("hidden");
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Predict";
  }
});

function showResult(data) {
  const pct = (data.probability * 100).toFixed(1);
  const badge = document.getElementById("badge");
  badge.textContent = `${data.risk_level} risk`;
  badge.className = `badge ${data.risk_level}`;

  document.getElementById("headline").textContent = data.dialysis_needed
    ? `Dialysis likely needed (${pct}% probability)`
    : `Dialysis unlikely needed (${pct}% probability)`;

  document.getElementById("barFill").style.width = `${pct}%`;
  document.getElementById("details").textContent =
    `GFR stage: ${data.gfr_stage}. ${data.disclaimer}`;

  resultBox.classList.remove("hidden");
}

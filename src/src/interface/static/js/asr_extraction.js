document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("asrExtractionForm");
  const submitBtn = document.getElementById("asrSubmitBtn");
  const languageSelect = document.getElementById("input_language");
  const audioFileInput = document.getElementById("audio_file");
  const startRecordingBtn = document.getElementById("startRecordingBtn");
  const stopRecordingBtn = document.getElementById("stopRecordingBtn");
  const recordingStatus = document.getElementById("recordingStatus");
  const recordingPreview = document.getElementById("recordingPreview");
  const translatedOutputCard = document.getElementById("translatedOutputCard");
  const translatedOutputStatus = document.getElementById("translatedOutputStatus");
  const translatedOutputText = document.getElementById("translatedOutputText");
  const previewCardTitle = document.getElementById("previewCardTitle");
  const translatedTextOverride = document.getElementById("translatedTextOverride");
  const rawTranscriptOverride = document.getElementById("rawTranscriptOverride");
  const numSpeakersHidden = document.getElementById("numSpeakersHidden");
  const modeUploadBtn = document.getElementById("modeUploadBtn");
  const modeRecordBtn = document.getElementById("modeRecordBtn");
  const uploadSection = document.getElementById("uploadSection");
  const recordSection = document.getElementById("recordSection");
  const enableDiarization = document.getElementById("enableDiarization");
  const diarizationOptions = document.getElementById("diarizationOptions");
  const numSpeakersInput = document.getElementById("numSpeakersInput");

  if (!form || !submitBtn || !languageSelect || !audioFileInput) return;

  let mediaRecorder = null;
  let mediaStream = null;
  let chunks = [];
  let selectedInputMode = "upload";

  // ── Speaker detection toggle ─────────────────────────────────────────────
  if (enableDiarization && diarizationOptions) {
    enableDiarization.addEventListener("change", () => {
      diarizationOptions.style.display = enableDiarization.checked ? "block" : "none";
      submitBtn.textContent = enableDiarization.checked
        ? "Detect Speakers & Run Extraction"
        : "Transcribe, Translate & Run Extraction";
    });
  }

  // ── Input-mode switching ─────────────────────────────────────────────────
  function clearSelectedAudioFile() {
    audioFileInput.value = "";
  }

  function setInputMode(mode) {
    selectedInputMode = mode;
    const usingUpload = mode === "upload";

    if (modeUploadBtn) {
      modeUploadBtn.classList.toggle("active", usingUpload);
      modeUploadBtn.setAttribute("aria-pressed", String(usingUpload));
    }
    if (modeRecordBtn) {
      modeRecordBtn.classList.toggle("active", !usingUpload);
      modeRecordBtn.setAttribute("aria-pressed", String(!usingUpload));
    }

    if (uploadSection) uploadSection.classList.toggle("asr-section-disabled", !usingUpload);
    if (recordSection) recordSection.classList.toggle("asr-section-disabled", usingUpload);

    if (usingUpload) {
      stopRecording();
      if (recordingStatus) recordingStatus.textContent = "Recording is disabled in file mode.";
      if (recordingPreview) {
        recordingPreview.style.display = "none";
        recordingPreview.removeAttribute("src");
      }
    } else {
      clearSelectedAudioFile();
      if (recordingStatus) recordingStatus.textContent = "Ready to record.";
    }
  }

  // ── Recording ────────────────────────────────────────────────────────────
  async function startRecording() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || typeof MediaRecorder === "undefined") {
      if (recordingStatus) recordingStatus.textContent = "Recording is not supported in this browser.";
      return;
    }

    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(mediaStream);
      chunks = [];

      mediaRecorder.addEventListener("dataavailable", (event) => {
        if (event.data && event.data.size > 0) chunks.push(event.data);
      });

      mediaRecorder.addEventListener("stop", () => {
        const mimeType = chunks[0]?.type || "audio/webm";
        const ext = mimeType.includes("ogg") ? "ogg" : "webm";
        const blob = new Blob(chunks, { type: mimeType });
        const file = new File([blob], `recording.${ext}`, { type: mimeType });

        const transfer = new DataTransfer();
        transfer.items.add(file);
        audioFileInput.files = transfer.files;

        if (recordingPreview) {
          recordingPreview.src = URL.createObjectURL(blob);
          recordingPreview.style.display = "block";
        }
        if (recordingStatus) recordingStatus.textContent = "Recording ready. You can submit now.";
      });

      mediaRecorder.start();
      if (recordingStatus) recordingStatus.textContent = "Recording...";
      if (startRecordingBtn) startRecordingBtn.disabled = true;
      if (stopRecordingBtn) stopRecordingBtn.disabled = false;
    } catch (error) {
      if (recordingStatus) {
        recordingStatus.textContent = `Could not start recording: ${error?.message || error}`;
      }
    }
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    }
    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => track.stop());
      mediaStream = null;
    }
    if (startRecordingBtn) startRecordingBtn.disabled = false;
    if (stopRecordingBtn) stopRecordingBtn.disabled = true;
  }

  if (startRecordingBtn && stopRecordingBtn) {
    startRecordingBtn.addEventListener("click", startRecording);
    stopRecordingBtn.addEventListener("click", stopRecording);
  }

  if (modeUploadBtn && modeRecordBtn) {
    modeUploadBtn.addEventListener("click", (event) => {
      event.preventDefault();
      setInputMode("upload");
    });
    modeRecordBtn.addEventListener("click", (event) => {
      event.preventDefault();
      setInputMode("record");
    });
    setInputMode("upload");
  }

  // ── Form submit ──────────────────────────────────────────────────────────
  form.addEventListener("submit", async (event) => {
    if (selectedInputMode === "upload" && (!audioFileInput.files || audioFileInput.files.length === 0)) {
      event.preventDefault();
      alert("Please upload an audio file before submitting.");
      return;
    }

    if (selectedInputMode === "record" && (!audioFileInput.files || audioFileInput.files.length === 0)) {
      event.preventDefault();
      alert("Please record audio and press Stop before submitting.");
      return;
    }

    event.preventDefault();

    const useDiarization = enableDiarization && enableDiarization.checked;
    const numSpeakers = numSpeakersInput ? parseInt(numSpeakersInput.value, 10) || 2 : 2;

    submitBtn.disabled = true;
    const languageLabel = languageSelect.options[languageSelect.selectedIndex]?.text || "selected language";

    // ── Diarization path ───────────────────────────────────────────────────
    if (useDiarization) {
      submitBtn.textContent = `Detecting speakers in ${languageLabel} audio…`;

      try {
        const previewBody = new FormData();
        previewBody.append("input_language", languageSelect.value);
        previewBody.append("num_speakers", String(numSpeakers));
        previewBody.append("audio_file", audioFileInput.files[0]);

        const previewRes = await fetch("/asr/diarize-preview", {
          method: "POST",
          body: previewBody,
        });

        const previewData = await previewRes.json();
        if (!previewRes.ok) {
          throw new Error(previewData.detail || previewData.error || "Speaker detection failed.");
        }

        const diarizedText = String(previewData.diarized_text || "").trim();
        const rawText = String(previewData.raw_text || "").trim();

        if (!diarizedText) {
          throw new Error("Diarization produced empty output.");
        }

        // Show preview
        if (translatedOutputCard) translatedOutputCard.style.display = "block";
        if (previewCardTitle) previewCardTitle.textContent = `Detected ${numSpeakers} Speaker(s) — Labelled Transcript`;
        if (translatedOutputText) translatedOutputText.value = diarizedText;
        if (translatedOutputStatus) {
          translatedOutputStatus.textContent =
            "Speaker-labelled transcript generated. Submitting for extraction…";
        }

        // Pass the diarized conversation text as override so the backend
        // skips re-transcription and uses speaker labels directly.
        if (translatedTextOverride) translatedTextOverride.value = diarizedText;
        if (rawTranscriptOverride) rawTranscriptOverride.value = rawText;
        // num_speakers = 0 tells backend to skip diarization (overrides already set)
        if (numSpeakersHidden) numSpeakersHidden.value = "0";

        const englishOption = Array.from(languageSelect.options).find((opt) => opt.value === "en");
        if (englishOption) languageSelect.value = "en";

        submitBtn.textContent = "Submitting speaker-labelled transcript…";
        setTimeout(() => form.submit(), 700);

      } catch (error) {
        submitBtn.disabled = false;
        submitBtn.textContent = "Detect Speakers & Run Extraction";
        alert(error?.message || String(error));
      }

      return;
    }

    // ── Standard single-speaker path ───────────────────────────────────────
    submitBtn.textContent = `Transcribing ${languageLabel} audio and translating…`;
    if (numSpeakersHidden) numSpeakersHidden.value = "0";

    try {
      const previewBody = new FormData();
      previewBody.append("input_language", languageSelect.value);
      previewBody.append("audio_file", audioFileInput.files[0]);

      const previewRes = await fetch("/asr/translate-preview", {
        method: "POST",
        body: previewBody,
      });

      const previewData = await previewRes.json();
      if (!previewRes.ok) {
        throw new Error(previewData.detail || previewData.error || "ASR translation failed.");
      }

      const translatedText = String(previewData.translated_text || "").trim();
      const rawText = String(previewData.raw_text || "").trim();
      if (!translatedText) {
        throw new Error("Translated text is empty.");
      }

      if (translatedOutputCard) translatedOutputCard.style.display = "block";
      if (previewCardTitle) previewCardTitle.textContent = "Translated Text (English)";
      if (translatedOutputText) translatedOutputText.value = translatedText;
      if (translatedOutputStatus) {
        translatedOutputStatus.textContent = "Translated text generated. Running extraction next...";
      }

      if (translatedTextOverride) translatedTextOverride.value = translatedText;
      if (rawTranscriptOverride) rawTranscriptOverride.value = rawText;

      const englishOption = Array.from(languageSelect.options).find((opt) => opt.value === "en");
      if (englishOption) languageSelect.value = "en";

      submitBtn.textContent = "Submitting translated text for extraction...";
      setTimeout(() => form.submit(), 700);
    } catch (error) {
      submitBtn.disabled = false;
      submitBtn.textContent = "Transcribe, Translate & Run Extraction";
      alert(error?.message || String(error));
    }
  });
});

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("asrExtractionForm");
  const submitBtn = document.getElementById("asrSubmitBtn");
  const languageSelect = document.getElementById("input_language");
  const audioFileInput = document.getElementById("audio_file");
  const startRecordingBtn = document.getElementById("startRecordingBtn");
  const stopRecordingBtn = document.getElementById("stopRecordingBtn");
  const recordingStatus = document.getElementById("recordingStatus");
  const recordingPreview = document.getElementById("recordingPreview");

  if (!form || !submitBtn || !languageSelect || !audioFileInput) return;

  let mediaRecorder = null;
  let mediaStream = null;
  let chunks = [];

  async function startRecording() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || typeof MediaRecorder === "undefined") {
      recordingStatus.textContent = "Recording is not supported in this browser.";
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

        recordingStatus.textContent = "Recording ready. You can submit now.";
      });

      mediaRecorder.start();
      recordingStatus.textContent = "Recording...";
      startRecordingBtn.disabled = true;
      stopRecordingBtn.disabled = false;
    } catch (error) {
      recordingStatus.textContent = `Could not start recording: ${error?.message || error}`;
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

    startRecordingBtn.disabled = false;
    stopRecordingBtn.disabled = true;
  }

  if (startRecordingBtn && stopRecordingBtn) {
    startRecordingBtn.addEventListener("click", startRecording);
    stopRecordingBtn.addEventListener("click", stopRecording);
  }

  form.addEventListener("submit", (event) => {
    if (!audioFileInput.files || audioFileInput.files.length === 0) {
      event.preventDefault();
      alert("Please upload an audio file or record audio before submitting.");
      return;
    }

    submitBtn.disabled = true;
    const languageLabel = languageSelect.options[languageSelect.selectedIndex]?.text || "selected language";
    submitBtn.textContent = `Transcribing ${languageLabel} audio, translating to English, then extracting...`;
  });
});

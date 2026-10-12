/**
 * PitchLab Application Controller
 * Orchestrates Video Slicer, Local CV Detection, 2D Radar, and Tactical Analytics
 */
(function() {
    'use strict';

    // Application Global State
    const state = {
        videoElement: null,
        videoDuration: 10,
        currentTime: 0,
        fps: 30,

        markerIn: 0,
        markerOut: 10,
        samplingInterval: 15, // frames

        extractedFrames: [],
        currentFrameIndex: 0,
        m2Playing: false,
        m2Speed: 1.0,
        m2PlayIntervalId: null,

        currentZonePreset: 'left_penalty_area',
        // Homography Calibration Quad (pixel coordinates on active frame)
        // Default initialized to left penalty area FIFA standard coordinates
        homographyPins: [
            { x: 75, y: 110, label: 'P1 (0, 13.8m)' },
            { x: 480, y: 95, label: 'P2 (16.5, 13.8m)' },
            { x: 550, y: 325, label: 'P3 (16.5, 54.2m)' },
            { x: 50, y: 335, label: 'P4 (0, 54.2m)' }
        ],
        destinationPitchPts: [
            { x: 0.0, y: 13.84 },
            { x: 16.5, y: 13.84 },
            { x: 16.5, y: 54.16 },
            { x: 0.0, y: 54.16 }
        ],
        homographyMatrix: null,
        draggedPinIndex: null,

        voronoiEnabled: false,
        demoIntervalId: null,
        isSyntheticDemo: false,

        // Backend integration
        videoFile: null,           // File object gdy user wybrał plik wideo
        backendAvailable: false,   // true gdy /api/videos odpowiada
        lastBackendVideoId: null,  // id z /api/upload (unikamy re-upload)
        lastBackendVideoFile: null // referencja do poprzednio uploadowanego File
    };

    // DOM Elements
    let els = {};

    function initElements() {
        els = {
            stepButtons: document.querySelectorAll('.step-btn'),
            moduleSections: document.querySelectorAll('.module-section'),

            videoInput: document.getElementById('video-file-input'),
            demoBtn: document.getElementById('btn-load-demo'),
            video: document.getElementById('video-player'),
            btnPlay: document.getElementById('btn-play-pause'),
            btnStepBack: document.getElementById('btn-step-back'),
            btnStepForward: document.getElementById('btn-step-forward'),
            timelineScrubber: document.getElementById('timeline-scrubber'),
            timeDisplay: document.getElementById('time-display'),

            inputMarkerIn: document.getElementById('input-marker-in'),
            inputMarkerOut: document.getElementById('input-marker-out'),
            btnSetIn: document.getElementById('btn-set-in'),
            btnSetOut: document.getElementById('btn-set-out'),
            intervalSlider: document.getElementById('interval-slider'),
            intervalValue: document.getElementById('interval-value'),
            framesCountEstimate: document.getElementById('frames-count-estimate'),

            btnExtractFrames: document.getElementById('btn-extract-frames'),
            extractionProgress: document.getElementById('extraction-progress'),
            extractionProgressBar: document.getElementById('extraction-progress-bar'),
            btnDownloadZip: document.getElementById('btn-download-zip'),
            btnGoModule2: document.getElementById('btn-go-module-2'),
            filmstripGallery: document.getElementById('filmstrip-gallery'),

            activeFrameCanvas: document.getElementById('active-frame-canvas'),
            pitchRadarCanvas: document.getElementById('pitch-radar-canvas'),
            zonePresetSelect: document.getElementById('zone-preset-select'),
            calibInstructionsText: document.getElementById('calib-instructions-text'),
            radarCameraFrustumBadge: document.getElementById('radar-camera-frustum-badge'),
            btnResetCalib: document.getElementById('btn-reset-calib'),
            btnProcessFrame: document.getElementById('btn-process-frame'),
            btnProcessAll: document.getElementById('btn-process-all'),
            btnGeminiAssist: document.getElementById('btn-gemini-assist'),
            btnExportJson: document.getElementById('btn-export-json'),
            btnExportCsv: document.getElementById('btn-export-csv'),
            tableBody: document.querySelector('#coordinates-table tbody'),

            // Module 1 presets
            btnFps1: document.getElementById('btn-fps-1'),
            btnFps5: document.getElementById('btn-fps-5'),
            btnFps15: document.getElementById('btn-fps-15'),
            btnFps30: document.getElementById('btn-fps-30'),

            // Module 2 playback
            m2BtnFirst: document.getElementById('m2-btn-first'),
            m2BtnPrev: document.getElementById('m2-btn-prev'),
            m2BtnPlay: document.getElementById('m2-btn-play'),
            m2BtnNext: document.getElementById('m2-btn-next'),
            m2BtnLast: document.getElementById('m2-btn-last'),
            m2FrameSlider: document.getElementById('m2-frame-slider'),
            m2FrameInfo: document.getElementById('m2-frame-info'),
            m2SpeedSelector: document.getElementById('m2-speed-selector'),

            // Module 3
            homeCentroidVal: document.getElementById('metric-home-centroid'),
            awayCentroidVal: document.getElementById('metric-away-centroid'),
            homeCompactVal: document.getElementById('metric-home-compactness'),
            awayCompactVal: document.getElementById('metric-away-compactness'),
            homeSpreadVal: document.getElementById('metric-home-spread'),
            awaySpreadVal: document.getElementById('metric-away-spread'),
            maxSpeedVal: document.getElementById('metric-max-speed'),
            btnToggleVoronoi: document.getElementById('btn-toggle-voronoi'),
            jsonSchemaViewer: document.getElementById('json-schema-viewer'),
            btnCopyJson: document.getElementById('btn-copy-json'),

            // Modal
            geminiModal: document.getElementById('gemini-modal'),
            btnCloseModal: document.getElementById('btn-close-modal')
        };
        state.videoElement = els.video;
    }

    // -------------------------------------------------------------
    // Navigation / Tabs
    // -------------------------------------------------------------
    function switchModule(stepNum) {
        els.stepButtons.forEach(btn => {
            btn.classList.toggle('active', parseInt(btn.dataset.step) === stepNum);
        });
        els.moduleSections.forEach(sec => {
            sec.classList.toggle('active', parseInt(sec.dataset.step) === stepNum);
        });

        if (stepNum === 2 || stepNum === 3) {
            renderActiveFrame();
            renderPitchRadar();
            updateTacticalMetrics();
        }
    }

    // -------------------------------------------------------------
    // Module 1: Video Player & Extraction Pipeline
    // -------------------------------------------------------------
    function setupVideoModule() {
        els.videoInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;
            stopSyntheticDemo();
            state.isSyntheticDemo = false;
            // Zapamiętaj plik do późniejszego uploadu do backendu
            state.videoFile = file;
            state.lastBackendVideoId = null;  // nowy plik = wymuś re-upload
            state.lastBackendVideoFile = null;
            const url = URL.createObjectURL(file);
            els.video.src = url;
            els.video.load();
        });

        els.demoBtn.addEventListener('click', () => {
            startSyntheticDemo();
        });

        els.video.addEventListener('loadedmetadata', () => {
            state.videoDuration = els.video.duration || 10;
            state.markerIn = 0;
            state.markerOut = Math.min(state.videoDuration, 10);
            els.timelineScrubber.max = state.videoDuration;
            els.inputMarkerIn.value = state.markerIn.toFixed(2);
            els.inputMarkerOut.value = state.markerOut.toFixed(2);
            updateEstimateCount();
            updateTimeDisplay();
        });

        els.video.addEventListener('timeupdate', () => {
            state.currentTime = els.video.currentTime;
            els.timelineScrubber.value = state.currentTime;
            updateTimeDisplay();
        });

        els.btnPlay.addEventListener('click', () => {
            if (els.video.paused) {
                els.video.play();
                els.btnPlay.innerHTML = '⏸ Pauza';
            } else {
                els.video.pause();
                els.btnPlay.innerHTML = '▶ Odtwórz';
            }
        });

        els.btnStepBack.addEventListener('click', () => {
            els.video.currentTime = Math.max(0, els.video.currentTime - (1 / state.fps));
        });

        els.btnStepForward.addEventListener('click', () => {
            els.video.currentTime = Math.min(state.videoDuration, els.video.currentTime + (1 / state.fps));
        });

        els.timelineScrubber.addEventListener('input', (e) => {
            els.video.currentTime = parseFloat(e.target.value);
        });

        els.btnSetIn.addEventListener('click', () => {
            state.markerIn = parseFloat(els.video.currentTime.toFixed(2));
            els.inputMarkerIn.value = state.markerIn.toFixed(2);
            updateEstimateCount();
        });

        els.btnSetOut.addEventListener('click', () => {
            state.markerOut = parseFloat(els.video.currentTime.toFixed(2));
            els.inputMarkerOut.value = state.markerOut.toFixed(2);
            updateEstimateCount();
        });

        els.inputMarkerIn.addEventListener('change', (e) => {
            state.markerIn = Math.max(0, parseFloat(e.target.value) || 0);
            updateEstimateCount();
        });

        els.inputMarkerOut.addEventListener('change', (e) => {
            state.markerOut = Math.min(state.videoDuration, parseFloat(e.target.value) || state.videoDuration);
            updateEstimateCount();
        });

        els.intervalSlider.addEventListener('input', (e) => {
            state.samplingInterval = parseInt(e.target.value);
            els.intervalValue.textContent = `${state.samplingInterval} klatek (~${(state.samplingInterval / state.fps).toFixed(2)}s)`;
            updateEstimateCount();
        });

        const setSampling = (interval) => {
            state.samplingInterval = interval;
            els.intervalSlider.value = interval;
            els.intervalValue.textContent = `${interval} klatek (~${(interval / state.fps).toFixed(2)}s)`;
            updateEstimateCount();
        };

        if (els.btnFps1) els.btnFps1.addEventListener('click', () => setSampling(1));
        if (els.btnFps5) els.btnFps5.addEventListener('click', () => setSampling(5));
        if (els.btnFps15) els.btnFps15.addEventListener('click', () => setSampling(15));
        if (els.btnFps30) els.btnFps30.addEventListener('click', () => setSampling(30));

        // Module 2 Clip Playback Controller
        if (els.m2BtnPlay) els.m2BtnPlay.addEventListener('click', togglePlayModule2Clip);
        if (els.m2BtnFirst) els.m2BtnFirst.addEventListener('click', () => selectFrame(0));
        if (els.m2BtnPrev) els.m2BtnPrev.addEventListener('click', () => selectFrame(state.currentFrameIndex - 1));
        if (els.m2BtnNext) els.m2BtnNext.addEventListener('click', () => selectFrame(state.currentFrameIndex + 1));
        if (els.m2BtnLast) els.m2BtnLast.addEventListener('click', () => selectFrame(state.extractedFrames.length - 1));

        if (els.m2FrameSlider) {
            els.m2FrameSlider.addEventListener('input', (e) => {
                selectFrame(parseInt(e.target.value) || 0);
            });
        }

        if (els.m2SpeedSelector) {
            els.m2SpeedSelector.addEventListener('change', (e) => {
                state.m2Speed = parseFloat(e.target.value) || 1.0;
                if (state.m2Playing) {
                    playModule2Clip();
                }
            });
        }

        els.btnExtractFrames.addEventListener('click', extractFramesPipeline);
        els.btnDownloadZip.addEventListener('click', downloadFramesZip);
        els.btnGoModule2.addEventListener('click', () => switchModule(2));
    }

    function updateTimeDisplay() {
        const cur = formatTime(state.currentTime);
        const dur = formatTime(state.videoDuration);
        els.timeDisplay.textContent = `${cur} / ${dur}`;
    }

    function updateEstimateCount() {
        const durationSec = Math.max(0, state.markerOut - state.markerIn);
        const totalFrames = durationSec * state.fps;
        const count = Math.max(1, Math.floor(totalFrames / state.samplingInterval) + 1);
        els.framesCountEstimate.textContent = count;
    }

    function formatTime(sec) {
        const m = Math.floor(sec / 60);
        const s = Math.floor(sec % 60);
        const ms = Math.floor((sec % 1) * 1000);
        return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(ms).padStart(3, '0')}`;
    }

    // -------------------------------------------------------------
    // Synthetic Match Action Demo Generator (Zero External Dep)
    // -------------------------------------------------------------
    function startSyntheticDemo() {
        stopSyntheticDemo();
        state.isSyntheticDemo = true;

        const offscreen = document.createElement('canvas');
        offscreen.width = 640;
        offscreen.height = 360;
        const octx = offscreen.getContext('2d');

        // Draw animated broadcast soccer scene to canvas
        let simTime = 0;
        const drawScene = () => {
            simTime += 0.05;

            // Grass field with green pitch texture & field lines
            octx.fillStyle = '#1e3a1e';
            octx.fillRect(0, 0, 640, 360);

            // Mowed grass stripes
            for (let i = 0; i < 640; i += 64) {
                octx.fillStyle = (i / 64) % 2 === 0 ? '#1b4d24' : '#1e5929';
                octx.fillRect(i, 40, 64, 280);
            }

            // Field boundary lines in perspective
            octx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
            octx.lineWidth = 2;
            octx.beginPath();
            octx.moveTo(60, 50); octx.lineTo(580, 50);
            octx.lineTo(620, 310); octx.lineTo(20, 310);
            octx.closePath();
            octx.stroke();

            // Center circle in perspective
            octx.beginPath();
            octx.ellipse(320, 180, 60, 24, 0, 0, Math.PI * 2);
            octx.stroke();
            octx.beginPath();
            octx.moveTo(320, 50); octx.lineTo(320, 310);
            octx.stroke();

            // Animated Home Players (Cyan)
            for (let i = 0; i < 7; i++) {
                const px = 100 + i * 32 + Math.sin(simTime + i) * 12;
                const py = 80 + (i % 3) * 60 + Math.cos(simTime * 0.8 + i) * 10;
                // Shadow
                octx.fillStyle = 'rgba(0,0,0,0.3)';
                octx.beginPath(); octx.ellipse(px + 6, py + 26, 8, 3, 0, 0, Math.PI * 2); octx.fill();
                // Player body
                octx.fillStyle = '#06B6D4';
                octx.fillRect(px, py, 14, 26);
                // Head
                octx.fillStyle = '#ffdbac';
                octx.beginPath(); octx.arc(px + 7, py - 4, 4, 0, Math.PI * 2); octx.fill();
            }

            // Animated Away Players (Coral)
            for (let i = 0; i < 7; i++) {
                const px = 360 + i * 30 - Math.sin(simTime + i) * 10;
                const py = 90 + (i % 3) * 62 + Math.cos(simTime * 0.9 + i) * 8;
                // Shadow
                octx.fillStyle = 'rgba(0,0,0,0.3)';
                octx.beginPath(); octx.ellipse(px + 6, py + 26, 8, 3, 0, 0, Math.PI * 2); octx.fill();
                // Player body
                octx.fillStyle = '#F43F5E';
                octx.fillRect(px, py, 14, 26);
                // Head
                octx.fillStyle = '#f1c27d';
                octx.beginPath(); octx.arc(px + 7, py - 4, 4, 0, Math.PI * 2); octx.fill();
            }

            // Referee
            octx.fillStyle = '#FACC15';
            octx.fillRect(310 + Math.sin(simTime * 0.5) * 8, 170, 12, 24);

            // Ball (Gold)
            const ballX = 300 + Math.sin(simTime * 1.5) * 60;
            const ballY = 175 + Math.cos(simTime * 1.5) * 30;
            octx.fillStyle = '#FFFFFF';
            octx.beginPath(); octx.arc(ballX, ballY, 4, 0, Math.PI * 2); octx.fill();
            octx.strokeStyle = '#F59E0B'; octx.stroke();
        };

        // Capture canvas stream to video element
        try {
            const stream = offscreen.captureStream(30);
            els.video.srcObject = stream;
            els.video.play();
            state.demoIntervalId = setInterval(drawScene, 1000 / 30);
            state.videoDuration = 12.0;
            state.markerIn = 0;
            state.markerOut = 8.0;
            els.inputMarkerIn.value = '0.00';
            els.inputMarkerOut.value = '8.00';
            updateEstimateCount();
        } catch (e) {
            console.warn('captureStream fallback to procedural frames', e);
        }
    }

    function stopSyntheticDemo() {
        if (state.demoIntervalId) {
            clearInterval(state.demoIntervalId);
            state.demoIntervalId = null;
        }
        if (els.video.srcObject) {
            els.video.srcObject = null;
        }
    }

    // -------------------------------------------------------------
    // Backend API: Upload + Analyze przez YOLO/ByteTrack pipeline
    // -------------------------------------------------------------

    /**
     * Uploaduje plik do backendu, tworzy segment i uruchamia analizę.
     * @param {File} videoFile
     * @param {number} startTime
     * @param {number} endTime
     * @returns {Promise<Object>} wynik analizy z backendu
     */
    async function analyzeSegmentViaBackend(videoFile, startTime, endTime) {
        // 1. Upload (tylko jeśli nowy plik)
        let videoId = state.lastBackendVideoId;
        if (!videoId || state.lastBackendVideoFile !== videoFile) {
            updateProgressLabel('Przesyłam wideo do backendu...');
            const formData = new FormData();
            formData.append('file', videoFile);
            const uploadResp = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            if (!uploadResp.ok) {
                const err = await uploadResp.text();
                throw new Error(`Upload failed (${uploadResp.status}): ${err}`);
            }
            const videoInfo = await uploadResp.json();
            videoId = videoInfo.id;
            state.lastBackendVideoId = videoId;
            state.lastBackendVideoFile = videoFile;
        }

        // 2. Utwórz segment
        updateProgressLabel('Tworzę segment w bazie...');
        const segResp = await fetch('/api/segments', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                video_id: videoId,
                start_time: startTime,
                end_time: endTime,
                label: 'Analiza CV (YOLO)'
            })
        });
        if (!segResp.ok) {
            const err = await segResp.text();
            throw new Error(`Segment creation failed (${segResp.status}): ${err}`);
        }
        const segment = await segResp.json();

        // 3. Uruchom analizę (długie - YOLO inference, timeout 180s)
        updateProgressLabel('⚡ Analizuję YOLO/ByteTrack (może chwilę trwać)...');
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 180000);
        try {
            const analyzeResp = await fetch(`/api/segments/${segment.id}/analyze`, {
                method: 'POST',
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            if (!analyzeResp.ok) {
                const err = await analyzeResp.text();
                throw new Error(`Analysis failed (${analyzeResp.status}): ${err}`);
            }
            return await analyzeResp.json();
        } catch (e) {
            clearTimeout(timeoutId);
            throw e;
        }
    }

    /**
     * Konwertuje wynik backendu na frameRecord.detection w formacie frontendowym.
     * @param {Object} apiResponse - wynik z /api/segments/{id}/analyze
     * @param {string[]} frameDataUrls - tablica dataURL thumbnailów (indeks = pozycja w frames[])
     * @returns {Object[]} tablica frameRecord
     */
    function backendFramesToDetections(apiResponse, frameDataUrls) {
        const isSynthetic = apiResponse.synthetic === true;
        const fps = apiResponse.fps || 25;
        const frames = apiResponse.frames || [];

        const results = [];
        let prevPlayers = [];
        let prevTimestamp = null;

        frames.forEach((frame, idx) => {
            const timestamp = frame.timestamp ?? (apiResponse.start_time + frame.frame_idx / fps);
            const dtSec = (prevTimestamp !== null) ? Math.max(0.001, timestamp - prevTimestamp) : (1 / fps);

            // Mapowanie playerów: filtruj niskie confidence, konwertuj pola
            const players = (frame.players || []).filter(p => {
                // Nie dodajemy zawodników o conf < 0.3 (chyba że syntetyk)
                return isSynthetic || (p.conf >= 0.3);
            }).map(p => {
                const trackId = 'P' + p.track_id;
                // Backend team: 'A'->home, 'B'->away, 'referee'->referee
                const team = p.team === 'A' ? 'home' : (p.team === 'B' ? 'away' : 'referee');
                // bbox backend: [x1,y1,x2,y2] -> frontend: [x1,y1,w,h]
                const bbox = Array.isArray(p.bbox) && p.bbox.length === 4
                    ? [p.bbox[0], p.bbox[1], p.bbox[2] - p.bbox[0], p.bbox[3] - p.bbox[1]]
                    : [0, 0, 20, 40];

                return {
                    trackId,
                    team,
                    jerseyColorHex: team === 'home' ? '#06B6D4' : (team === 'referee' ? '#FACC15' : '#F43F5E'),
                    bbox,
                    pitchX: p.x,
                    pitchY: p.y,
                    rawX: p.x,
                    rawY: p.y,
                    inPitch: (p.x >= 0 && p.x <= 105 && p.y >= 0 && p.y <= 68),
                    velocityX: 0,
                    velocityY: 0,
                    speedKmh: 0,
                    conf: p.conf,
                    _fromBackend: true,
                    _synthetic: isSynthetic
                };
            });

            // Przelicz prędkości między klatkami (tylko dla wiarygodnych detekcji)
            players.forEach(curr => {
                const prev = prevPlayers.find(p => p.trackId === curr.trackId);
                if (prev && curr.conf >= 0.3 && prev.conf >= 0.3) {
                    const vx = (curr.pitchX - prev.pitchX) / dtSec;
                    const vy = (curr.pitchY - prev.pitchY) / dtSec;
                    const speed = Math.hypot(vx, vy);
                    // Limit fizyczny: zawodnik max ~43 km/h
                    if (speed * 3.6 <= 45) {
                        curr.velocityX = Math.round(vx * 100) / 100;
                        curr.velocityY = Math.round(vy * 100) / 100;
                        curr.speedKmh = Math.round(speed * 3.6 * 10) / 10;
                    }
                }
            });

            // Piłka
            let ball = { frameX: 0, frameY: 0, pitchX: 0, pitchY: 0, confidence: 0, isDetected: false, inPitch: false };
            if (frame.ball && frame.ball.conf >= 0.3) {
                const b = frame.ball;
                const bboxCenter = Array.isArray(b.bbox) && b.bbox.length === 4
                    ? [(b.bbox[0] + b.bbox[2]) / 2, (b.bbox[1] + b.bbox[3]) / 2]
                    : [0, 0];
                ball = {
                    frameX: bboxCenter[0],
                    frameY: bboxCenter[1],
                    pitchX: b.x,
                    pitchY: b.y,
                    confidence: b.conf,
                    isDetected: true,
                    inPitch: (b.x >= 0 && b.x <= 105 && b.y >= 0 && b.y <= 68),
                    _fromBackend: true
                };
            }

            const frameRecord = {
                frameIndex: frame.frame_idx,
                timestampSeconds: timestamp,
                timecode: formatTime(timestamp),
                dataUrl: frameDataUrls[idx] || '',
                detection: {
                    frameIndex: frame.frame_idx,
                    timestampSeconds: timestamp,
                    timecode: formatTime(timestamp),
                    dimensions: { width: 640, height: 360 },
                    pitchDimensions: { lengthMeters: 105, widthMeters: 68 },
                    players,
                    ball,
                    _fromBackend: true,
                    _synthetic: isSynthetic
                },
                _fromBackend: true,
                _synthetic: isSynthetic
            };

            results.push(frameRecord);
            prevPlayers = players;
            prevTimestamp = timestamp;
        });

        return results;
    }

    function updateProgressLabel(text) {
        if (els.extractionProgressBar) {
            const label = els.extractionProgressBar.parentElement;
            if (label) {
                const span = label.querySelector('.progress-label');
                if (span) span.textContent = text;
            }
        }
        console.info('[PitchLab]', text);
    }

    // -------------------------------------------------------------
    // Frame Extractor (Canvas Frame Grabber + Backend YOLO)
    // -------------------------------------------------------------
    async function extractFramesPipeline() {
        els.btnExtractFrames.disabled = true;
        els.extractionProgress.style.display = 'block';
        els.filmstripGallery.innerHTML = '';
        state.extractedFrames = [];

        const dtSec = state.samplingInterval / state.fps;
        const startSec = state.markerIn;
        const endSec = state.markerOut;
        const sampleTimes = [];

        for (let t = startSec; t <= endSec; t += dtSec) {
            sampleTimes.push(parseFloat(t.toFixed(3)));
        }
        if (sampleTimes.length === 0) sampleTimes.push(startSec);

        // --- Próba użycia backendu (YOLO) ---
        const useBackend = state.backendAvailable && !state.isSyntheticDemo && state.videoFile;
        let backendResult = null;

        if (useBackend) {
            try {
                backendResult = await analyzeSegmentViaBackend(state.videoFile, startSec, endSec);
            } catch (err) {
                console.warn('[PitchLab] Backend YOLO niedostępny, przełączam na CV heurystyczny:', err.message);
                backendResult = null;
                // Pokaż ostrzeżenie w UI
                updateProgressLabel('⚠️ Fallback: CV heurystyczny (YOLO niedostępny)');
            }
        }

        // Zawsze robimy canvas capture dla thumbnailów
        const captureCanvas = document.createElement('canvas');
        captureCanvas.width = 640;
        captureCanvas.height = 360;
        const cctx = captureCanvas.getContext('2d');

        updateHomography();

        if (backendResult && backendResult.frames && backendResult.frames.length > 0) {
            // === TRYB BACKEND: Pobierz thumbnails i użyj danych z YOLO ===
            updateProgressLabel('⚡ Pobieranie klatek wideo (thumbnails)...');
            const dataUrls = [];

            // Budujemy mapę: timestamp -> dataUrl
            const backendFrames = backendResult.frames;
            const backendTimestamps = backendFrames.map(f =>
                f.timestamp ?? (backendResult.start_time + f.frame_idx / (backendResult.fps || 25))
            );

            for (let i = 0; i < backendFrames.length; i++) {
                const timePoint = backendTimestamps[i];

                try {
                    els.video.currentTime = timePoint;
                    await new Promise(res => {
                        const onSeeked = () => { els.video.removeEventListener('seeked', onSeeked); res(); };
                        els.video.addEventListener('seeked', onSeeked);
                        setTimeout(res, 500);  // timeout zabezpieczający
                    });
                    cctx.drawImage(els.video, 0, 0, 640, 360);
                } catch (_) {
                    cctx.fillStyle = '#1e3a1e';
                    cctx.fillRect(0, 0, 640, 360);
                }

                dataUrls.push(captureCanvas.toDataURL('image/jpeg', 0.75));

                const pct = Math.round(((i + 1) / backendFrames.length) * 50 + 50);
                els.extractionProgressBar.style.width = `${pct}%`;
            }

            // Konwertuj wyniki backendu na format frontendowy
            const frameRecords = backendFramesToDetections(backendResult, dataUrls);

            frameRecords.forEach((frameRecord, i) => {
                state.extractedFrames.push(frameRecord);
                appendThumbnailCard(frameRecord, i);
            });

            const syntheticNote = backendResult.synthetic ? ' (dane SYNTETYCZNE - brak YOLO)' : '';
            console.info(`[PitchLab] Backend analiza OK: ${frameRecords.length} klatek${syntheticNote}`);
            if (backendResult.synthetic) {
                updateProgressLabel('⚠️ Dane syntetyczne (backend nie mógł uruchomić YOLO)');
            } else {
                updateProgressLabel(`✅ YOLO: ${frameRecords.length} klatek, średnio ${backendResult.stats?.tracked_players_count || '?'} zawodników`);
            }

        } else {
            // === TRYB FALLBACK: Stary heurystyczny CV (analyzeFrame) ===
            if (!state.isSyntheticDemo && state.videoFile) {
                updateProgressLabel('🔵 Tryb offline: CV heurystyczny (bez YOLO)');
            }
            let prevPlayers = [];

            for (let i = 0; i < sampleTimes.length; i++) {
                const timePoint = sampleTimes[i];
                const frameIndex = Math.round(timePoint * state.fps);

                if (!state.isSyntheticDemo) {
                    els.video.currentTime = timePoint;
                    await new Promise(res => {
                        const onSeeked = () => {
                            els.video.removeEventListener('seeked', onSeeked);
                            res();
                        };
                        els.video.addEventListener('seeked', onSeeked);
                    });
                }

                try {
                    cctx.drawImage(els.video, 0, 0, captureCanvas.width, captureCanvas.height);
                } catch (err) {
                    cctx.fillStyle = '#1e3a1e';
                    cctx.fillRect(0, 0, captureCanvas.width, captureCanvas.height);
                }

                const dataUrl = captureCanvas.toDataURL('image/jpeg', 0.85);

                const detection = window.PitchLabCV.Detector.analyzeFrame(
                    cctx,
                    captureCanvas.width,
                    captureCanvas.height,
                    state.homographyMatrix,
                    prevPlayers,
                    frameIndex,
                    timePoint
                );

                window.PitchLabCV.Tactics.calcVelocities(detection.players, prevPlayers, dtSec);
                prevPlayers = detection.players;

                const frameRecord = {
                    frameIndex,
                    timestampSeconds: timePoint,
                    timecode: formatTime(timePoint),
                    dataUrl,
                    detection
                };
                state.extractedFrames.push(frameRecord);
                appendThumbnailCard(frameRecord, i);

                const pct = Math.round(((i + 1) / sampleTimes.length) * 100);
                els.extractionProgressBar.style.width = `${pct}%`;
            }
        }

        els.btnExtractFrames.disabled = false;
        els.btnDownloadZip.disabled = false;
        els.btnGoModule2.disabled = false;

        if (state.extractedFrames.length > 0) {
            selectFrame(0);
        }
    }

    function appendThumbnailCard(frameRecord, idx) {
        const card = document.createElement('div');
        card.className = `frame-thumb-card ${idx === 0 ? 'active' : ''}`;
        card.dataset.index = idx;

        card.innerHTML = `
            <img src="${frameRecord.dataUrl}" alt="Klatka #${frameRecord.frameIndex}" />
            <div class="frame-card-meta">
                <span class="frame-badge">#${frameRecord.frameIndex}</span>
                <span class="frame-time">${frameRecord.timecode}</span>
            </div>
        `;

        card.addEventListener('click', () => {
            selectFrame(idx);
        });

        els.filmstripGallery.appendChild(card);
    }

    function selectFrame(idx) {
        if (!state.extractedFrames || state.extractedFrames.length === 0) return;
        idx = Math.max(0, Math.min(idx, state.extractedFrames.length - 1));
        state.currentFrameIndex = idx;
        document.querySelectorAll('.frame-thumb-card').forEach((c, i) => {
            c.classList.toggle('active', i === idx);
        });
        updateModule2PlayerUI();
        renderActiveFrame();
        renderPitchRadar();
        updateTacticalMetrics();
    }

    function updateModule2PlayerUI() {
        const total = state.extractedFrames.length;
        if (!els.m2FrameSlider) return;
        els.m2FrameSlider.max = Math.max(0, total - 1);
        els.m2FrameSlider.value = state.currentFrameIndex;
        if (total === 0) {
            if (els.m2FrameInfo) els.m2FrameInfo.textContent = 'Klatka: 0 / 0 (00:00.000)';
        } else {
            const cur = state.extractedFrames[state.currentFrameIndex];
            if (els.m2FrameInfo && cur) {
                els.m2FrameInfo.textContent = `Klatka: ${state.currentFrameIndex + 1} / ${total} (${cur.timecode})`;
            }
        }
        if (els.m2BtnPlay) {
            els.m2BtnPlay.textContent = state.m2Playing ? '⏸ Pauza (2D)' : '▶ Odtwarzaj wideo/radar 2D';
        }
    }

    function playModule2Clip() {
        if (!state.extractedFrames || state.extractedFrames.length <= 1) return;
        state.m2Playing = true;
        updateModule2PlayerUI();
        if (state.m2PlayIntervalId) clearInterval(state.m2PlayIntervalId);
        const dtPerFrame = (state.samplingInterval / state.fps) * 1000;
        const intervalMs = Math.max(25, (dtPerFrame || 100) / state.m2Speed);
        state.m2PlayIntervalId = setInterval(() => {
            let next = state.currentFrameIndex + 1;
            if (next >= state.extractedFrames.length) {
                next = 0; // loop playback
            }
            selectFrame(next);
        }, intervalMs);
    }

    function pauseModule2Clip() {
        state.m2Playing = false;
        if (state.m2PlayIntervalId) {
            clearInterval(state.m2PlayIntervalId);
            state.m2PlayIntervalId = null;
        }
        updateModule2PlayerUI();
    }

    function togglePlayModule2Clip() {
        if (state.m2Playing) {
            pauseModule2Clip();
        } else {
            playModule2Clip();
        }
    }

    // -------------------------------------------------------------
    // JSZip Frame Exporter (ZIP + manifest.json)
    // -------------------------------------------------------------
    async function downloadFramesZip() {
        if (!state.extractedFrames || state.extractedFrames.length === 0) {
            alert('Brak wyekstrahowanych klatek do pobrania.');
            return;
        }

        if (typeof window.JSZip === 'undefined') {
            alert('Biblioteka JSZip nie jest jeszcze załadowana.');
            return;
        }

        const zip = new window.JSZip();
        const framesFolder = zip.folder('frames');

        const manifest = {
            project: 'PitchLab - Soccer Video Spatial Analytics',
            exportedAt: new Date().toISOString(),
            totalFrames: state.extractedFrames.length,
            pitchDimensions: { lengthMeters: 105, widthMeters: 68 },
            frames: []
        };

        state.extractedFrames.forEach((frame, idx) => {
            const fileName = `frame_${String(frame.frameIndex).padStart(5, '0')}.jpg`;
            const base64Data = frame.dataUrl.split(',')[1];
            framesFolder.file(fileName, base64Data, { base64: true });

            manifest.frames.push({
                fileName,
                frameIndex: frame.frameIndex,
                timestampSeconds: frame.timestampSeconds,
                timecode: frame.timecode,
                playerCount: frame.detection.players.length,
                ball: frame.detection.ball
            });
        });

        zip.file('manifest.json', JSON.stringify(manifest, null, 2));

        const blob = await zip.generateAsync({ type: 'blob' });
        const downloadUrl = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = downloadUrl;
        a.download = `pitchlab_frames_${Date.now()}.zip`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(downloadUrl);
    }

    // -------------------------------------------------------------
    // Module 2: Homography Calibrator & Dual Canvas Workspace
    // -------------------------------------------------------------
    function updateHomography() {
        state.homographyMatrix = window.PitchLabCV.Homography.solve(
            state.homographyPins,
            state.destinationPitchPts
        );
    }

    function setupCalibratorEvents() {
        const canvas = els.activeFrameCanvas;

        const getPos = (e) => {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            return {
                x: (e.clientX - rect.left) * scaleX,
                y: (e.clientY - rect.top) * scaleY
            };
        };

        canvas.addEventListener('mousedown', (e) => {
            const pos = getPos(e);
            for (let i = 0; i < state.homographyPins.length; i++) {
                const pin = state.homographyPins[i];
                if (Math.hypot(pin.x - pos.x, pin.y - pos.y) <= 18) {
                    state.draggedPinIndex = i;
                    break;
                }
            }
        });

        window.addEventListener('mousemove', (e) => {
            if (state.draggedPinIndex === null) return;
            const pos = getPos(e);
            state.homographyPins[state.draggedPinIndex].x = Math.max(5, Math.min(canvas.width - 5, pos.x));
            state.homographyPins[state.draggedPinIndex].y = Math.max(5, Math.min(canvas.height - 5, pos.y));
            updateHomography();
            renderActiveFrame();
            reprojectCurrentFrame();
            renderPitchRadar();
            updateTacticalMetrics();
        });

        window.addEventListener('mouseup', () => {
            state.draggedPinIndex = null;
        });

        function applyZonePreset(zoneKey) {
            state.currentZonePreset = zoneKey;
            const presets = window.PitchLabCV.Homography.zonePresets;
            const preset = presets ? presets[zoneKey] : null;
            if (!preset) return;

            state.homographyPins = JSON.parse(JSON.stringify(preset.pins));
            state.destinationPitchPts = JSON.parse(JSON.stringify(preset.dstPts));

            if (els.calibInstructionsText) {
                els.calibInstructionsText.textContent = preset.instructions;
            }
            if (els.zonePresetSelect && els.zonePresetSelect.value !== zoneKey) {
                els.zonePresetSelect.value = zoneKey;
            }

            updateHomography();
            renderActiveFrame();
            reprojectCurrentFrame();
            renderPitchRadar();
            updateTacticalMetrics();
        }

        if (els.zonePresetSelect) {
            els.zonePresetSelect.addEventListener('change', (e) => {
                applyZonePreset(e.target.value);
            });
        }

        els.btnResetCalib.addEventListener('click', () => {
            applyZonePreset(state.currentZonePreset || 'left_penalty_area');
        });

        els.btnProcessFrame.addEventListener('click', () => {
            if (state.extractedFrames.length > 0) {
                const cur = state.extractedFrames[state.currentFrameIndex];
                const off = document.createElement('canvas');
                off.width = 640; off.height = 360;
                const ctx = off.getContext('2d');
                const img = new Image();
                img.onload = () => {
                    ctx.drawImage(img, 0, 0);
                    cur.detection = window.PitchLabCV.Detector.analyzeFrame(
                        ctx, 640, 360, state.homographyMatrix, [], cur.frameIndex, cur.timestampSeconds
                    );
                    renderActiveFrame();
                    renderPitchRadar();
                    updateTacticalMetrics();
                };
                img.src = cur.dataUrl;
            }
        });

        els.btnExportJson.addEventListener('click', exportMatchTrackingJson);
        els.btnExportCsv.addEventListener('click', exportPositionsCsv);

        // Gemini Flash Modal
        els.btnGeminiAssist.addEventListener('click', () => {
            els.geminiModal.classList.add('active');
        });
        els.btnCloseModal.addEventListener('click', () => {
            els.geminiModal.classList.remove('active');
        });
    }

    function reprojectCurrentFrame() {
        if (!state.extractedFrames || state.extractedFrames.length === 0) return;
        const cur = state.extractedFrames[state.currentFrameIndex];
        if (!cur || !cur.detection) return;

        cur.detection.players.forEach(p => {
            const footX = p.bbox[0] + p.bbox[2] / 2;
            const footY = p.bbox[1] + p.bbox[3];
            const proj = window.PitchLabCV.Homography.project(state.homographyMatrix, footX, footY);
            p.pitchX = proj.pitchX;
            p.pitchY = proj.pitchY;
            p.rawX = proj.rawX;
            p.rawY = proj.rawY;
            p.inPitch = proj.inPitch;
        });

        if (cur.detection.ball && cur.detection.ball.isDetected) {
            const ballProj = window.PitchLabCV.Homography.project(
                state.homographyMatrix,
                cur.detection.ball.frameX,
                cur.detection.ball.frameY
            );
            cur.detection.ball.pitchX = ballProj.pitchX;
            cur.detection.ball.pitchY = ballProj.pitchY;
            cur.detection.ball.inPitch = ballProj.inPitch;
        }
    }

    function renderActiveFrame() {
        const canvas = els.activeFrameCanvas;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const cur = state.extractedFrames[state.currentFrameIndex];
        if (cur && cur.dataUrl) {
            const img = new Image();
            img.onload = () => {
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                drawDetectionOverlays(ctx, cur.detection);
                drawHomographyHandles(ctx);
            };
            img.src = cur.dataUrl;
        } else {
            // Draw default empty state
            ctx.fillStyle = '#101728';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = '#64748B';
            ctx.font = '13px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Wyodrębnij klatki w Module 1, aby zobaczyć podgląd', canvas.width / 2, canvas.height / 2);
            drawHomographyHandles(ctx);
        }
    }

    function drawDetectionOverlays(ctx, detection) {
        if (!detection) return;

        // Draw Player BBoxes & Badges
        detection.players.forEach(p => {
            const [x, y, w, h] = p.bbox;

            // Box outline
            ctx.strokeStyle = p.team === 'home' ? '#06B6D4' : p.team === 'away' ? '#F43F5E' : '#FACC15';
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, w, h);

            // Foot contact marker
            ctx.fillStyle = '#10B981';
            ctx.beginPath();
            ctx.arc(x + w / 2, y + h, 4, 0, Math.PI * 2);
            ctx.fill();

            // Label
            ctx.fillStyle = ctx.strokeStyle;
            ctx.font = '10px JetBrains Mono, monospace';
            ctx.fillText(`${p.trackId} (${p.pitchX.toFixed(1)}m)`, x, Math.max(12, y - 4));
        });

        // Draw Ball
        if (detection.ball && detection.ball.isDetected) {
            ctx.strokeStyle = '#F59E0B';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(detection.ball.frameX, detection.ball.frameY, 8, 0, Math.PI * 2);
            ctx.stroke();
            ctx.fillStyle = '#FFFFFF';
            ctx.beginPath();
            ctx.arc(detection.ball.frameX, detection.ball.frameY, 4, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    function drawHomographyHandles(ctx) {
        const pins = state.homographyPins;

        // Draw connecting quadrilateral
        ctx.strokeStyle = 'rgba(16, 185, 129, 0.7)';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(pins[0].x, pins[0].y);
        ctx.lineTo(pins[1].x, pins[1].y);
        ctx.lineTo(pins[2].x, pins[2].y);
        ctx.lineTo(pins[3].x, pins[3].y);
        ctx.closePath();
        ctx.stroke();
        ctx.setLineDash([]);

        // Draw draggable pin handles
        pins.forEach((pin, i) => {
            ctx.fillStyle = '#10B981';
            ctx.beginPath();
            ctx.arc(pin.x, pin.y, 8, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#FFFFFF';
            ctx.lineWidth = 2;
            ctx.stroke();

            ctx.fillStyle = '#FFFFFF';
            ctx.font = '10px sans-serif';
            ctx.fillText(pin.label, pin.x + 12, pin.y + 4);
        });
    }

    // -------------------------------------------------------------
    // 2D Pitch Radar Canvas Rendering (105m x 68m FIFA Standard)
    // -------------------------------------------------------------
    function renderPitchRadar() {
        const canvas = els.pitchRadarCanvas;
        const ctx = canvas.getContext('2d');
        const W = canvas.width;
        const H = canvas.height;

        ctx.clearRect(0, 0, W, H);

        // Grass pitch background
        ctx.fillStyle = '#0f381e';
        ctx.fillRect(0, 0, W, H);

        // Mowed pitch stripes
        for (let i = 0; i < W; i += W / 10) {
            ctx.fillStyle = (Math.floor(i / (W / 10)) % 2 === 0) ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.03)';
            ctx.fillRect(i, 0, W / 10, H);
        }

        const scaleX = W / 105;
        const scaleY = H / 68;

        // Render Voronoi space control if enabled
        const cur = state.extractedFrames[state.currentFrameIndex];
        if (state.voronoiEnabled && cur && cur.detection) {
            window.PitchLabCV.Tactics.renderVoronoi(ctx, W, H, cur.detection.players);
        }

        // Pitch Markings in Crisp White
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.75)';
        ctx.lineWidth = 2;

        // Touchlines
        ctx.strokeRect(0, 0, W, H);

        // Halfway line
        ctx.beginPath();
        ctx.moveTo(W / 2, 0); ctx.lineTo(W / 2, H);
        ctx.stroke();

        // Center circle (radius 9.15m)
        ctx.beginPath();
        ctx.arc(W / 2, H / 2, 9.15 * scaleX, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = 'rgba(255,255,255,0.7)';
        ctx.beginPath();
        ctx.arc(W / 2, H / 2, 3, 0, Math.PI * 2);
        ctx.fill();

        // Penalty Boxes (16.5m depth, 40.32m width)
        const boxDepthPx = 16.5 * scaleX;
        const boxHeightPx = 40.32 * scaleY;
        const boxY = (H - boxHeightPx) / 2;
        ctx.strokeRect(0, boxY, boxDepthPx, boxHeightPx);
        ctx.strokeRect(W - boxDepthPx, boxY, boxDepthPx, boxHeightPx);

        // Goal Areas (5.5m depth, 18.32m width)
        const goalDepthPx = 5.5 * scaleX;
        const goalHeightPx = 18.32 * scaleY;
        const goalY = (H - goalHeightPx) / 2;
        ctx.strokeRect(0, goalY, goalDepthPx, goalHeightPx);
        ctx.strokeRect(W - goalDepthPx, goalY, goalDepthPx, goalHeightPx);

        // Goals (2.44m x 7.32m)
        const goalMouthH = 7.32 * scaleY;
        const goalMouthY = (H - goalMouthH) / 2;
        ctx.strokeRect(-4, goalMouthY, 4, goalMouthH);
        ctx.strokeRect(W, goalMouthY, 4, goalMouthH);

        // Draw Camera Field of View (FOV Frustum) on 2D Pitch Radar
        if (state.homographyMatrix) {
            // Test ground-plane corners below frame horizon
            const testCorners = [
                { x: 0, y: 70 },
                { x: 640, y: 70 },
                { x: 640, y: 360 },
                { x: 0, y: 360 }
            ];

            const fov = testCorners.map(pt => window.PitchLabCV.Homography.project(state.homographyMatrix, pt.x, pt.y));
            const allValid = fov.every(pt => pt.validProjection);

            if (allValid) {
                ctx.save();
                ctx.fillStyle = 'rgba(56, 189, 248, 0.14)';
                ctx.strokeStyle = '#38BDF8';
                ctx.lineWidth = 1.5;
                ctx.setLineDash([5, 4]);

                ctx.beginPath();
                const startX = Math.max(0, Math.min(105, fov[0].pitchX)) * scaleX;
                const startY = Math.max(0, Math.min(68, fov[0].pitchY)) * scaleY;
                ctx.moveTo(startX, startY);
                for (let i = 1; i < fov.length; i++) {
                    const cx = Math.max(0, Math.min(105, fov[i].pitchX)) * scaleX;
                    const cy = Math.max(0, Math.min(68, fov[i].pitchY)) * scaleY;
                    ctx.lineTo(cx, cy);
                }
                ctx.closePath();
                ctx.fill();
                ctx.stroke();
                ctx.restore();

                // Calculate visible field span & action sector
                const validPoints = fov.filter(pt => pt.inPitch);
                if (validPoints.length > 0) {
                    const minX = Math.min(...validPoints.map(pt => pt.pitchX));
                    const maxX = Math.max(...validPoints.map(pt => pt.pitchX));
                    const spanX = Math.max(1, Math.round((maxX - minX) * 10) / 10);
                    const midX = (minX + maxX) / 2;

                    let sectorName = 'Środek Boiska';
                    if (midX < 35) sectorName = 'Lewe Pole Karne (Bramka L)';
                    else if (midX > 70) sectorName = 'Prawe Pole Karne (Bramka P)';

                    if (els.radarCameraFrustumBadge) {
                        els.radarCameraFrustumBadge.textContent = `📹 Kadr: ${sectorName} (~${spanX}m pola)`;
                    }
                }
            } else if (els.radarCameraFrustumBadge) {
                els.radarCameraFrustumBadge.textContent = `📹 Kadr: Zbliżenie (Horyzont kamery poza boiskiem)`;
            }
        }

        // Draw Players on 2D Radar
        if (cur && cur.detection) {
            cur.detection.players.forEach(p => {
                // Clamp coordinates ONLY for radar display drawing
                const drawX = Math.max(0, Math.min(105, p.pitchX)) * scaleX;
                const drawY = Math.max(0, Math.min(68, p.pitchY)) * scaleY;

                // Player dot
                ctx.fillStyle = p.team === 'home' ? '#06B6D4' : p.team === 'away' ? '#F43F5E' : '#FACC15';
                ctx.beginPath();
                ctx.arc(drawX, drawY, 6, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = '#FFFFFF';
                ctx.lineWidth = 1.5;
                ctx.stroke();

                // Track ID
                ctx.fillStyle = '#FFFFFF';
                ctx.font = '9px JetBrains Mono, monospace';
                ctx.fillText(p.trackId, drawX + 8, drawY + 3);

                // Physical Velocity Arrow: 1 second vector in meters, capped at 12 m/s (~43 km/h)
                const speed = Math.hypot(p.velocityX || 0, p.velocityY || 0);
                if (speed > 0.1 && speed <= 12.0) {
                    const dispLengthMeters = Math.min(speed, 6.0);
                    const normVx = (p.velocityX || 0) / speed;
                    const normVy = (p.velocityY || 0) / speed;
                    const arrowEndX = drawX + (normVx * dispLengthMeters) * scaleX;
                    const arrowEndY = drawY + (normVy * dispLengthMeters) * scaleY;

                    ctx.strokeStyle = ctx.fillStyle;
                    ctx.lineWidth = 1.5;
                    ctx.beginPath();
                    ctx.moveTo(drawX, drawY);
                    ctx.lineTo(arrowEndX, arrowEndY);
                    ctx.stroke();
                }
            });

            // Ball on Radar - strictly only when honestly detected
            if (cur.detection.ball && cur.detection.ball.isDetected) {
                const drawBx = Math.max(0, Math.min(105, cur.detection.ball.pitchX)) * scaleX;
                const drawBy = Math.max(0, Math.min(68, cur.detection.ball.pitchY)) * scaleY;

                ctx.fillStyle = '#F59E0B';
                ctx.beginPath();
                ctx.arc(drawBx, drawBy, 5, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = '#FFFFFF';
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }

            // Update numerical data table
            populateDataTable(cur.detection);
        }
    }

    function populateDataTable(detection) {
        els.tableBody.innerHTML = '';
        if (!detection) return;

        detection.players.forEach(p => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>${p.trackId}</strong></td>
                <td><span class="tag-badge tag-${p.team}">${p.team}</span></td>
                <td><span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:${p.jerseyColorHex};vertical-align:middle;margin-right:6px;"></span>${p.jerseyColorHex}</td>
                <td>[${p.bbox.join(', ')}]</td>
                <td>${(p.bbox[0] + p.bbox[2] / 2).toFixed(0)}, ${(p.bbox[1] + p.bbox[3]).toFixed(0)}</td>
                <td><strong>${p.pitchX.toFixed(2)}m, ${p.pitchY.toFixed(2)}m</strong></td>
                <td>${p.speedKmh ? p.speedKmh.toFixed(1) + ' km/h' : '0.0 km/h'}</td>
            `;
            els.tableBody.appendChild(tr);
        });

        if (detection.ball) {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>PIŁKA</strong></td>
                <td><span class="tag-badge tag-ball">BALL</span></td>
                <td>#FFFFFF</td>
                <td>-</td>
                <td>${detection.ball.frameX}, ${detection.ball.frameY}</td>
                <td><strong>${detection.ball.pitchX.toFixed(2)}m, ${detection.ball.pitchY.toFixed(2)}m</strong></td>
                <td>-</td>
            `;
            els.tableBody.appendChild(tr);
        }
    }

    // -------------------------------------------------------------
    // Module 3: Tactical Metrics Dashboard & Contract
    // -------------------------------------------------------------
    function updateTacticalMetrics() {
        const cur = state.extractedFrames[state.currentFrameIndex];
        if (!cur || !cur.detection) return;

        const metrics = window.PitchLabCV.Tactics.teamMetrics(cur.detection.players);

        els.homeCentroidVal.textContent = `${metrics.home.centroid.x}m, ${metrics.home.centroid.y}m`;
        els.awayCentroidVal.textContent = `${metrics.away.centroid.x}m, ${metrics.away.centroid.y}m`;
        els.homeCompactVal.textContent = `${metrics.home.compactness}m`;
        els.awayCompactVal.textContent = `${metrics.away.compactness}m`;
        els.homeSpreadVal.textContent = `Szer: ${metrics.home.width}m | Dł: ${metrics.home.length}m`;
        els.awaySpreadVal.textContent = `Szer: ${metrics.away.width}m | Dł: ${metrics.away.length}m`;

        let maxSpd = 0;
        cur.detection.players.forEach(p => {
            if (p.speedKmh && p.speedKmh > maxSpd) maxSpd = p.speedKmh;
        });
        els.maxSpeedVal.textContent = `${maxSpd.toFixed(1)} km/h`;

        // Update JSON Schema Viewer
        els.jsonSchemaViewer.textContent = JSON.stringify(cur.detection, null, 2);
    }

    function setupModule3Events() {
        els.btnToggleVoronoi.addEventListener('click', () => {
            state.voronoiEnabled = !state.voronoiEnabled;
            els.btnToggleVoronoi.textContent = state.voronoiEnabled
                ? 'Wyłącz mapę Voronoi'
                : 'Włącz mapę kontroli przestrzeni (Voronoi)';
            renderPitchRadar();
        });

        els.btnCopyJson.addEventListener('click', () => {
            navigator.clipboard.writeText(els.jsonSchemaViewer.textContent)
                .then(() => alert('JSON skopiowany do schowka!'))
                .catch(() => alert('Nie udało się skopiować JSON.'));
        });
    }

    // -------------------------------------------------------------
    // Data Exporters (JSON & CSV)
    // -------------------------------------------------------------
    function exportMatchTrackingJson() {
        if (!state.extractedFrames || state.extractedFrames.length === 0) {
            alert('Brak danych do wyeksportowania.');
            return;
        }

        const payload = {
            exportTimestamp: new Date().toISOString(),
            pitchDimensions: { lengthMeters: 105, widthMeters: 68 },
            totalFrames: state.extractedFrames.length,
            frames: state.extractedFrames.map(f => f.detection)
        };

        const jsonStr = JSON.stringify(payload, null, 2);
        downloadFile(jsonStr, 'match_tracking_data.json', 'application/json');
    }

    function exportPositionsCsv() {
        if (!state.extractedFrames || state.extractedFrames.length === 0) {
            alert('Brak danych do wyeksportowania.');
            return;
        }

        let csv = 'frameIndex,timecode,trackId,team,jerseyColorHex,bbox_x,bbox_y,bbox_w,bbox_h,pitchX_m,pitchY_m,speedKmh\n';

        state.extractedFrames.forEach(f => {
            const det = f.detection;
            det.players.forEach(p => {
                csv += `${det.frameIndex},${det.timecode},${p.trackId},${p.team},${p.jerseyColorHex},${p.bbox[0]},${p.bbox[1]},${p.bbox[2]},${p.bbox[3]},${p.pitchX},${p.pitchY},${p.speedKmh || 0}\n`;
            });
            if (det.ball) {
                csv += `${det.frameIndex},${det.timecode},BALL,ball,#FFFFFF,${det.ball.frameX},${det.ball.frameY},0,0,${det.ball.pitchX},${det.ball.pitchY},0\n`;
            }
        });

        downloadFile(csv, 'positions.csv', 'text/csv');
    }

    function downloadFile(content, fileName, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = fileName;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // -------------------------------------------------------------
    // Backend Health Check
    // -------------------------------------------------------------
    async function checkBackendStatus() {
        const badge = document.getElementById('backend-status-badge');
        try {
            const resp = await fetch('/api/videos', { signal: AbortSignal.timeout(3000) });
            if (resp.ok) {
                state.backendAvailable = true;
                if (badge) {
                    badge.textContent = '🟢 Backend aktywny (YOLO)';
                    badge.className = 'backend-badge backend-online';
                }
                console.info('[PitchLab] Backend dostępny – tryb YOLO aktywny');
            } else {
                throw new Error('Backend HTTP ' + resp.status);
            }
        } catch (_) {
            state.backendAvailable = false;
            if (badge) {
                badge.textContent = '🔴 Tryb offline (CV heurystyczny)';
                badge.className = 'backend-badge backend-offline';
            }
            console.warn('[PitchLab] Backend niedostępny – fallback do CV heurystycznego');
        }
    }

    // -------------------------------------------------------------
    // Initialization
    // -------------------------------------------------------------
    function init() {
        initElements();
        updateHomography();
        updateModule2PlayerUI();

        // Nav click events
        els.stepButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                switchModule(parseInt(btn.dataset.step));
            });
        });

        setupVideoModule();
        setupCalibratorEvents();
        setupModule3Events();

        // Sprawdź status backendu asynchronicznie (nie blokuje UI)
        checkBackendStatus();

        // Initial empty canvas renders
        renderActiveFrame();
        renderPitchRadar();
    }

    document.addEventListener('DOMContentLoaded', init);
})();

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

        // Homography Calibration Quad (pixel coordinates on active frame)
        // Default initialized relative to canvas size
        homographyPins: [
            { x: 80, y: 60, label: 'TL (0, 0)' },
            { x: 560, y: 60, label: 'TR (105, 0)' },
            { x: 600, y: 320, label: 'BR (105, 68)' },
            { x: 40, y: 320, label: 'BL (0, 68)' }
        ],
        destinationPitchPts: [
            { x: 0, y: 0 },
            { x: 105, y: 0 },
            { x: 105, y: 68 },
            { x: 0, y: 68 }
        ],
        homographyMatrix: null,
        draggedPinIndex: null,

        voronoiEnabled: false,
        demoIntervalId: null,
        isSyntheticDemo: false
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
    // Frame Extractor (Canvas Frame Grabber)
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

        const captureCanvas = document.createElement('canvas');
        captureCanvas.width = 640;
        captureCanvas.height = 360;
        const cctx = captureCanvas.getContext('2d');

        updateHomography();
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

            // Draw video to capture canvas
            try {
                cctx.drawImage(els.video, 0, 0, captureCanvas.width, captureCanvas.height);
            } catch (err) {
                // If CORS prevents drawing external video, fill with styled placeholder
                cctx.fillStyle = '#1e3a1e';
                cctx.fillRect(0, 0, captureCanvas.width, captureCanvas.height);
            }

            const dataUrl = captureCanvas.toDataURL('image/jpeg', 0.85);

            // Execute in-browser CV extraction
            const detection = window.PitchLabCV.Detector.analyzeFrame(
                cctx,
                captureCanvas.width,
                captureCanvas.height,
                state.homographyMatrix,
                prevPlayers,
                frameIndex,
                timePoint
            );

            // Compute velocities between frames
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

            // Append Thumbnail Card to Filmstrip
            appendThumbnailCard(frameRecord, i);

            // Progress bar update
            const pct = Math.round(((i + 1) / sampleTimes.length) * 100);
            els.extractionProgressBar.style.width = `${pct}%`;
        }

        els.btnExtractFrames.disabled = false;
        els.btnDownloadZip.disabled = false;
        els.btnGoModule2.disabled = false;

        // Select first frame by default
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

        els.btnResetCalib.addEventListener('click', () => {
            state.homographyPins = [
                { x: 80, y: 60, label: 'TL (0, 0)' },
                { x: 560, y: 60, label: 'TR (105, 0)' },
                { x: 600, y: 320, label: 'BR (105, 68)' },
                { x: 40, y: 320, label: 'BL (0, 68)' }
            ];
            updateHomography();
            renderActiveFrame();
            reprojectCurrentFrame();
            renderPitchRadar();
            updateTacticalMetrics();
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
        });

        const ballProj = window.PitchLabCV.Homography.project(
            state.homographyMatrix,
            cur.detection.ball.frameX,
            cur.detection.ball.frameY
        );
        cur.detection.ball.pitchX = ballProj.pitchX;
        cur.detection.ball.pitchY = ballProj.pitchY;
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

        // Draw Players on 2D Radar
        if (cur && cur.detection) {
            cur.detection.players.forEach(p => {
                const px = p.pitchX * scaleX;
                const py = p.pitchY * scaleY;

                // Player dot
                ctx.fillStyle = p.team === 'home' ? '#06B6D4' : p.team === 'away' ? '#F43F5E' : '#FACC15';
                ctx.beginPath();
                ctx.arc(px, py, 6, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = '#FFFFFF';
                ctx.lineWidth = 1.5;
                ctx.stroke();

                // Track ID
                ctx.fillStyle = '#FFFFFF';
                ctx.font = '9px JetBrains Mono, monospace';
                ctx.fillText(p.trackId, px + 8, py + 3);

                // Velocity Arrow
                if (p.velocityX || p.velocityY) {
                    ctx.strokeStyle = ctx.fillStyle;
                    ctx.beginPath();
                    ctx.moveTo(px, py);
                    ctx.lineTo(px + p.velocityX * 3, py + p.velocityY * 3);
                    ctx.stroke();
                }
            });

            // Ball on Radar
            if (cur.detection.ball) {
                const bx = cur.detection.ball.pitchX * scaleX;
                const by = cur.detection.ball.pitchY * scaleY;

                ctx.fillStyle = '#F59E0B';
                ctx.beginPath();
                ctx.arc(bx, by, 5, 0, Math.PI * 2);
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

        // Initial empty canvas renders
        renderActiveFrame();
        renderPitchRadar();
    }

    document.addEventListener('DOMContentLoaded', init);
})();

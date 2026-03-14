import { useEffect, useMemo, useRef, useState } from 'react'

const DEFAULT_API = 'http://localhost:8000'
const SETTINGS_KEY = 'levee-ui-settings-v1'
const TARGET_CATALOG = [
  { id: 'sandboil', label: 'Sandboil' },
  { id: 'seepage', label: 'Seepage' },
  { id: 'rutting', label: 'Rutting' },
  { id: 'crack', label: 'Crack' },
  { id: 'potholes', label: 'Potholes' },
  { id: 'encroachment', label: 'Encroachment' },
  { id: 'animal_burrow', label: 'Animal Burrow' },
  { id: 'vegetation', label: 'Vegetation' },
]

function defaultThreshold(model) {
  return model === 'seepage' ? 0.98 : 0.5
}

function toPointString(points) {
  return points.map((p) => `${p.x},${p.y}`).join(' ')
}

function InteractiveNetworkBackground() {
  const canvasRef = useRef(null)
  const nodesRef = useRef([])
  const mouseRef = useRef({ x: -5000, y: -5000 })
  const rafRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')

    function randomNode(w, h, near) {
      const baseX = near ? near.x : Math.random() * w
      const baseY = near ? near.y : Math.random() * h
      return {
        x: baseX + (Math.random() - 0.5) * 80,
        y: baseY + (Math.random() - 0.5) * 80,
        vx: (Math.random() - 0.5) * 0.7,
        vy: (Math.random() - 0.5) * 0.7,
        r: 1.3 + Math.random() * 2.2,
      }
    }

    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      canvas.width = Math.floor(window.innerWidth * dpr)
      canvas.height = Math.floor(window.innerHeight * dpr)
      canvas.style.width = `${window.innerWidth}px`
      canvas.style.height = `${window.innerHeight}px`
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

      const target = Math.max(36, Math.min(95, Math.floor((window.innerWidth * window.innerHeight) / 28000)))
      if (!nodesRef.current.length) {
        nodesRef.current = Array.from({ length: target }, () => randomNode(window.innerWidth, window.innerHeight))
      } else if (nodesRef.current.length < target) {
        const add = target - nodesRef.current.length
        for (let i = 0; i < add; i += 1) {
          nodesRef.current.push(randomNode(window.innerWidth, window.innerHeight))
        }
      } else if (nodesRef.current.length > target) {
        nodesRef.current = nodesRef.current.slice(0, target)
      }
    }

    function onMove(e) {
      mouseRef.current = { x: e.clientX, y: e.clientY }
    }

    function onLeave() {
      mouseRef.current = { x: -5000, y: -5000 }
    }

    function onClick(e) {
      const near = { x: e.clientX, y: e.clientY }
      for (let i = 0; i < 7; i += 1) {
        nodesRef.current.push(randomNode(window.innerWidth, window.innerHeight, near))
      }
      if (nodesRef.current.length > 140) {
        nodesRef.current = nodesRef.current.slice(nodesRef.current.length - 140)
      }
    }

    function tick() {
      const w = window.innerWidth
      const h = window.innerHeight

      ctx.clearRect(0, 0, w, h)

      const nodes = nodesRef.current
      const mouse = mouseRef.current

      for (let i = 0; i < nodes.length; i += 1) {
        const n = nodes[i]

        const dxm = n.x - mouse.x
        const dym = n.y - mouse.y
        const dMouse = Math.hypot(dxm, dym)

        if (dMouse < 160) {
          const repel = (160 - dMouse) / 1400
          n.vx += (dxm / (dMouse + 0.001)) * repel
          n.vy += (dym / (dMouse + 0.001)) * repel
        }

        n.vx *= 0.992
        n.vy *= 0.992

        n.x += n.vx
        n.y += n.vy

        if (n.x < 0 || n.x > w) n.vx *= -1
        if (n.y < 0 || n.y > h) n.vy *= -1

        n.x = Math.max(0, Math.min(w, n.x))
        n.y = Math.max(0, Math.min(h, n.y))
      }

      for (let i = 0; i < nodes.length; i += 1) {
        const a = nodes[i]
        for (let j = i + 1; j < nodes.length; j += 1) {
          const b = nodes[j]
          const d = Math.hypot(a.x - b.x, a.y - b.y)
          if (d < 120) {
            const alpha = (120 - d) / 120
            ctx.strokeStyle = `rgba(45, 212, 191, ${alpha * 0.28})`
            ctx.lineWidth = 1
            ctx.beginPath()
            ctx.moveTo(a.x, a.y)
            ctx.lineTo(b.x, b.y)
            ctx.stroke()
          }
        }

        const dm = Math.hypot(a.x - mouse.x, a.y - mouse.y)
        if (dm < 180) {
          const alpha = (180 - dm) / 180
          ctx.strokeStyle = `rgba(56, 189, 248, ${alpha * 0.45})`
          ctx.lineWidth = 1.1
          ctx.beginPath()
          ctx.moveTo(a.x, a.y)
          ctx.lineTo(mouse.x, mouse.y)
          ctx.stroke()
        }
      }

      for (let i = 0; i < nodes.length; i += 1) {
        const n = nodes[i]
        ctx.fillStyle = 'rgba(8, 145, 178, 0.85)'
        ctx.beginPath()
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2)
        ctx.fill()
      }

      rafRef.current = requestAnimationFrame(tick)
    }

    resize()
    window.addEventListener('resize', resize)
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseleave', onLeave)
    window.addEventListener('click', onClick)
    rafRef.current = requestAnimationFrame(tick)

    return () => {
      window.removeEventListener('resize', resize)
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseleave', onLeave)
      window.removeEventListener('click', onClick)
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [])

  return <canvas className="network-canvas" ref={canvasRef} aria-hidden="true" />
}

export default function App() {
  const [apiBase, setApiBase] = useState(DEFAULT_API)
  const [models, setModels] = useState([])
  const [selectedModels, setSelectedModels] = useState([])
  const [thresholds, setThresholds] = useState({})
  const [thresholdTypes, setThresholdTypes] = useState({})
  const [overlayIntensity, setOverlayIntensity] = useState(0.45)
  const [distanceThreshold, setDistanceThreshold] = useState(20)
  const [visualization, setVisualization] = useState('overlay')
  const [showPreprocessPreview, setShowPreprocessPreview] = useState(false)
  const [autoRun, setAutoRun] = useState(true)
  const [activeTab, setActiveTab] = useState('detection')

  const [resolutionFactor, setResolutionFactor] = useState(1.0)
  const [brightnessFactor, setBrightnessFactor] = useState(0)
  const [contrastFactor, setContrastFactor] = useState(0)
  const [blurAmount, setBlurAmount] = useState(1)
  const [edgeDetection, setEdgeDetection] = useState(false)
  const [flipHorizontal, setFlipHorizontal] = useState(false)
  const [flipVertical, setFlipVertical] = useState(false)
  const [rotateAngle, setRotateAngle] = useState(0)

  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [lastInferenceMs, setLastInferenceMs] = useState(null)
  const [lastFromCache, setLastFromCache] = useState(false)
  const [inferenceCount, setInferenceCount] = useState(0)

  const [hitlEnabled, setHitlEnabled] = useState(false)
  const [annotationTarget, setAnnotationTarget] = useState('sandboil')
  const [annotationPoints, setAnnotationPoints] = useState([])
  const [annotationNotes, setAnnotationNotes] = useState('')
  const [annotationMessage, setAnnotationMessage] = useState('')
  const [savedAnnotations, setSavedAnnotations] = useState([])

  const abortRef = useRef(null)
  const outputImgRef = useRef(null)
  const inferenceCacheRef = useRef(new Map())

  function applyPreset(name) {
    if (name === 'baseline') {
      setResolutionFactor(1.0)
      setBrightnessFactor(0)
      setContrastFactor(0)
      setBlurAmount(1)
      setEdgeDetection(false)
      setFlipHorizontal(false)
      setFlipVertical(false)
      setRotateAngle(0)
      return
    }

    if (name === 'sharp-edges') {
      setResolutionFactor(1.1)
      setBrightnessFactor(10)
      setContrastFactor(20)
      setBlurAmount(1)
      setEdgeDetection(true)
      setFlipHorizontal(false)
      setFlipVertical(false)
      setRotateAngle(0)
      return
    }

    if (name === 'noise-suppression') {
      setResolutionFactor(1.0)
      setBrightnessFactor(0)
      setContrastFactor(5)
      setBlurAmount(7)
      setEdgeDetection(false)
      setFlipHorizontal(false)
      setFlipVertical(false)
      setRotateAngle(0)
    }
  }

  function resetAll() {
    setOverlayIntensity(0.45)
    setDistanceThreshold(20)
    setVisualization('overlay')
    setShowPreprocessPreview(false)
    setAutoRun(true)
    setActiveTab('detection')
    applyPreset('baseline')
    inferenceCacheRef.current.clear()
    setLastInferenceMs(null)
    setLastFromCache(false)
    setInferenceCount(0)
  }

  useEffect(() => {
    try {
      const raw = localStorage.getItem(SETTINGS_KEY)
      if (!raw) return
      const saved = JSON.parse(raw)

      if (typeof saved.apiBase === 'string') setApiBase(saved.apiBase)
      if (typeof saved.overlayIntensity === 'number') setOverlayIntensity(saved.overlayIntensity)
      if (typeof saved.distanceThreshold === 'number') setDistanceThreshold(saved.distanceThreshold)
      if (typeof saved.visualization === 'string') setVisualization(saved.visualization)
      if (typeof saved.showPreprocessPreview === 'boolean') setShowPreprocessPreview(saved.showPreprocessPreview)
      if (typeof saved.autoRun === 'boolean') setAutoRun(saved.autoRun)
      if (typeof saved.activeTab === 'string') setActiveTab(saved.activeTab)

      if (typeof saved.resolutionFactor === 'number') setResolutionFactor(saved.resolutionFactor)
      if (typeof saved.brightnessFactor === 'number') setBrightnessFactor(saved.brightnessFactor)
      if (typeof saved.contrastFactor === 'number') setContrastFactor(saved.contrastFactor)
      if (typeof saved.blurAmount === 'number') setBlurAmount(saved.blurAmount)
      if (typeof saved.edgeDetection === 'boolean') setEdgeDetection(saved.edgeDetection)
      if (typeof saved.flipHorizontal === 'boolean') setFlipHorizontal(saved.flipHorizontal)
      if (typeof saved.flipVertical === 'boolean') setFlipVertical(saved.flipVertical)
      if (typeof saved.rotateAngle === 'number') setRotateAngle(saved.rotateAngle)
    } catch {
      // ignore malformed saved settings
    }
  }, [])

  useEffect(() => {
    const snapshot = {
      apiBase,
      overlayIntensity,
      distanceThreshold,
      visualization,
      showPreprocessPreview,
      autoRun,
      activeTab,
      resolutionFactor,
      brightnessFactor,
      contrastFactor,
      blurAmount,
      edgeDetection,
      flipHorizontal,
      flipVertical,
      rotateAngle,
    }
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(snapshot))
  }, [
    apiBase,
    overlayIntensity,
    distanceThreshold,
    visualization,
    showPreprocessPreview,
    autoRun,
    activeTab,
    resolutionFactor,
    brightnessFactor,
    contrastFactor,
    blurAmount,
    edgeDetection,
    flipHorizontal,
    flipVertical,
    rotateAngle,
  ])

  useEffect(() => {
    async function loadModels() {
      try {
        const resp = await fetch(`${apiBase}/models`)
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
        const data = await resp.json()
        const available = data.available_models || []
        setModels(available)

        if (available.length) {
          const preferred = available.includes('sandboil') ? 'sandboil' : available[0]
          setSelectedModels([preferred])
          setAnnotationTarget(preferred)
          const initThresholds = {}
          const initTypes = {}
          available.forEach((m) => {
            initThresholds[m] = defaultThreshold(m)
            initTypes[m] = 'Manual'
          })
          setThresholds(initThresholds)
          setThresholdTypes(initTypes)
        }
      } catch (err) {
        setError(`Could not fetch models: ${err.message}`)
      }
    }
    loadModels()
  }, [apiBase])

  useEffect(() => {
    if (selectedModels.length && !selectedModels.includes(annotationTarget)) {
      setAnnotationTarget(selectedModels[0])
    }
  }, [selectedModels, annotationTarget])

  const modelOptions = useMemo(() => {
    const catalogById = new Map(TARGET_CATALOG.map((item) => [item.id, item]))
    const fromCatalog = TARGET_CATALOG.map((item) => ({
      ...item,
      available: models.includes(item.id),
    }))

    const unknownAvailable = models
      .filter((m) => !catalogById.has(m))
      .map((m) => ({ id: m, label: m, available: true }))

    return [...fromCatalog, ...unknownAvailable]
  }, [models])

  async function refreshAnnotations() {
    try {
      const resp = await fetch(`${apiBase}/annotations?limit=10`)
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const data = await resp.json()
      setSavedAnnotations(data.items || [])
    } catch {
      // non-blocking
    }
  }

  useEffect(() => {
    refreshAnnotations()
  }, [apiBase])

  const outputImage = useMemo(() => {
    if (!result?.image_base64) return ''
    return `data:image/png;base64,${result.image_base64}`
  }, [result])

  const preprocessedImage = useMemo(() => {
    if (!result?.preprocessed_base64) return ''
    return `data:image/png;base64,${result.preprocessed_base64}`
  }, [result])

  async function runInference() {
    if (!file) {
      setError('Please choose an image first.')
      return
    }
    if (!selectedModels.length) {
      setError('Please select at least one model.')
      return
    }

    setLoading(true)
    setError('')
    setLastFromCache(false)

    try {
      const payloadSnapshot = {
        apiBase,
        selectedModels,
        thresholds,
        thresholdTypes,
        visualization,
        overlayIntensity,
        distanceThreshold,
        preprocess: {
          resolutionFactor,
          brightnessFactor,
          contrastFactor,
          blurAmount,
          edgeDetection,
          flipHorizontal,
          flipVertical,
          rotateAngle,
        },
        fileMeta: {
          name: file.name,
          size: file.size,
          lastModified: file.lastModified,
        },
      }
      const cacheKey = JSON.stringify(payloadSnapshot)
      if (inferenceCacheRef.current.has(cacheKey)) {
        setResult(inferenceCacheRef.current.get(cacheKey))
        setLastFromCache(true)
        setLastInferenceMs(0)
        setInferenceCount((v) => v + 1)
        setLoading(false)
        return
      }

      if (abortRef.current) {
        abortRef.current.abort()
      }
      const controller = new AbortController()
      abortRef.current = controller
      const startedAt = performance.now()

      const formData = new FormData()
      formData.append('image', file)
      formData.append('model_type', selectedModels[0])
      formData.append('selected_models', JSON.stringify(selectedModels))
      formData.append('thresholds', JSON.stringify(thresholds))
      formData.append('threshold_types', JSON.stringify(thresholdTypes))
      formData.append('visualization', visualization)
      formData.append('overlay_intensity', String(overlayIntensity))
      formData.append('distance_threshold', String(distanceThreshold))
      formData.append(
        'preprocessing_settings',
        JSON.stringify({
          resolution_factor: resolutionFactor,
          brightness_factor: brightnessFactor,
          contrast_factor: contrastFactor,
          blur_amount: blurAmount,
          edge_detection: edgeDetection,
          flip_horizontal: flipHorizontal,
          flip_vertical: flipVertical,
          rotate_angle: rotateAngle,
        }),
      )

      const resp = await fetch(`${apiBase}/infer/image`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      })

      const data = await resp.json()
      if (!resp.ok) {
        throw new Error(data.detail || `HTTP ${resp.status}`)
      }

      setResult(data)
      setLastInferenceMs(Math.round(performance.now() - startedAt))
      setInferenceCount((v) => v + 1)

      inferenceCacheRef.current.set(cacheKey, data)
      if (inferenceCacheRef.current.size > 30) {
        const oldest = inferenceCacheRef.current.keys().next().value
        inferenceCacheRef.current.delete(oldest)
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(err.message)
      }
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (!autoRun || !file || !selectedModels.length) {
      return
    }

    const timer = setTimeout(() => {
      runInference()
    }, 450)

    return () => {
      clearTimeout(timer)
      if (abortRef.current) {
        abortRef.current.abort()
      }
    }
  }, [
    autoRun,
    file,
    apiBase,
    selectedModels,
    thresholds,
    thresholdTypes,
    overlayIntensity,
    distanceThreshold,
    visualization,
    resolutionFactor,
    brightnessFactor,
    contrastFactor,
    blurAmount,
    edgeDetection,
    flipHorizontal,
    flipVertical,
    rotateAngle,
  ])

  function onImageClick(event) {
    if (!outputImgRef.current || !hitlEnabled) return

    const rect = outputImgRef.current.getBoundingClientRect()
    const relX = event.clientX - rect.left
    const relY = event.clientY - rect.top
    const scaleX = outputImgRef.current.naturalWidth / rect.width
    const scaleY = outputImgRef.current.naturalHeight / rect.height

    const x = Math.max(0, Math.round(relX * scaleX))
    const y = Math.max(0, Math.round(relY * scaleY))
    setAnnotationPoints((prev) => [...prev, { x, y }])
  }

  async function saveAnnotation() {
    if (!annotationPoints || annotationPoints.length < 3) {
      setAnnotationMessage('Add at least 3 points to save annotation.')
      return
    }

    try {
      setAnnotationMessage('Saving annotation...')
      const resp = await fetch(`${apiBase}/annotations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          target_model: annotationTarget,
          image_name: file?.name || null,
          points: annotationPoints,
          notes: annotationNotes || null,
          metadata: {
            selected_models: selectedModels,
            thresholds,
            threshold_types: thresholdTypes,
          },
        }),
      })

      const data = await resp.json()
      if (!resp.ok) {
        throw new Error(data.detail || `HTTP ${resp.status}`)
      }

      setAnnotationMessage(`Saved annotation ${data.id}`)
      setAnnotationPoints([])
      setAnnotationNotes('')
      refreshAnnotations()
    } catch (err) {
      setAnnotationMessage(`Save failed: ${err.message}`)
    }
  }

  return (
    <>
      <InteractiveNetworkBackground />

      <div className="app-shell">
        <header className="hero">
          <h1>Levee Detection Console</h1>
        </header>

        <main className="grid">
          <section className="panel controls">
            <h2>Control Center</h2>

            <div className="tab-row">
              <button className={activeTab === 'detection' ? 'tab-btn active' : 'tab-btn'} onClick={() => setActiveTab('detection')}>Detection</button>
              <button className={activeTab === 'visual' ? 'tab-btn active' : 'tab-btn'} onClick={() => setActiveTab('visual')}>Visual</button>
              <button className={activeTab === 'preprocess' ? 'tab-btn active' : 'tab-btn'} onClick={() => setActiveTab('preprocess')}>Preprocess</button>
              <button className={activeTab === 'run' ? 'tab-btn active' : 'tab-btn'} onClick={() => setActiveTab('run')}>Run</button>
            </div>

            <div className="tab-panel">
              {activeTab === 'detection' && (
                <>
                  <label>API Base URL</label>
                  <input value={apiBase} onChange={(e) => setApiBase(e.target.value)} />

                  <h3>Detection Targets</h3>
                  <div className="check-grid">
                    {modelOptions.map((m) => (
                      <label key={m.id} className={`check-item ${m.available ? '' : 'unavailable'}`}>
                        <input
                          type="radio"
                          name="detection-target"
                          checked={selectedModels[0] === m.id}
                          disabled={!m.available}
                          onChange={() => {
                            if (!m.available) {
                              setError(`${m.label} model is coming soon.`)
                              return
                            }
                            setError('')
                            setSelectedModels([m.id])
                            setAnnotationTarget(m.id)
                          }}
                        />
                        <span>{m.available ? m.label : `${m.label} (Coming Soon)`}</span>
                      </label>
                    ))}
                  </div>

                  {selectedModels.map((m) => (
                    <div key={m} className="model-box">
                      <strong>{m}</strong>
                      <label>Threshold Mode</label>
                      <select
                        value={thresholdTypes[m] || 'Manual'}
                        onChange={(e) => setThresholdTypes((prev) => ({ ...prev, [m]: e.target.value }))}
                      >
                        <option value="Manual">Manual</option>
                        <option value="Automatic">Automatic (Otsu)</option>
                      </select>
                      {(thresholdTypes[m] || 'Manual') === 'Manual' && (
                        <>
                          <label>Threshold: {(thresholds[m] ?? defaultThreshold(m)).toFixed(2)}</label>
                          <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.01"
                            value={thresholds[m] ?? defaultThreshold(m)}
                            onChange={(e) =>
                              setThresholds((prev) => ({ ...prev, [m]: Number(e.target.value) }))
                            }
                          />
                        </>
                      )}
                    </div>
                  ))}
                </>
              )}

              {activeTab === 'visual' && (
                <>
                  <label>Output Mode</label>
                  <select value={visualization} onChange={(e) => setVisualization(e.target.value)}>
                    <option value="overlay">Overlay</option>
                    <option value="bbox">Bounding Box</option>
                  </select>

                  <label>Overlay Intensity: {overlayIntensity.toFixed(2)}</label>
                  <input type="range" min="0" max="1" step="0.01" value={overlayIntensity} onChange={(e) => setOverlayIntensity(Number(e.target.value))} />

                  <label>Distance Threshold (overlap): {distanceThreshold}px</label>
                  <input type="range" min="5" max="100" step="1" value={distanceThreshold} onChange={(e) => setDistanceThreshold(Number(e.target.value))} />

                  <div className="check-grid">
                    <label className="check-item"><input type="checkbox" checked={showPreprocessPreview} onChange={(e) => setShowPreprocessPreview(e.target.checked)} /><span>Show Preprocessed View</span></label>
                  </div>
                </>
              )}

              {activeTab === 'preprocess' && (
                <>
                  <h3>Quick Presets</h3>
                  <div className="preset-row">
                    <button type="button" className="preset-btn" onClick={() => applyPreset('baseline')}>Baseline</button>
                    <button type="button" className="preset-btn" onClick={() => applyPreset('sharp-edges')}>Sharp Edges</button>
                    <button type="button" className="preset-btn" onClick={() => applyPreset('noise-suppression')}>Noise Suppression</button>
                  </div>

                  <label>Resolution Scale: {resolutionFactor.toFixed(2)}</label>
                  <input type="range" min="0.2" max="2" step="0.1" value={resolutionFactor} onChange={(e) => setResolutionFactor(Number(e.target.value))} />

                  <label>Brightness: {brightnessFactor}</label>
                  <input type="range" min="-100" max="100" step="1" value={brightnessFactor} onChange={(e) => setBrightnessFactor(Number(e.target.value))} />

                  <label>Contrast: {contrastFactor}</label>
                  <input type="range" min="-100" max="100" step="1" value={contrastFactor} onChange={(e) => setContrastFactor(Number(e.target.value))} />

                  <label>Gaussian Blur Kernel: {blurAmount}</label>
                  <input type="range" min="1" max="15" step="2" value={blurAmount} onChange={(e) => setBlurAmount(Number(e.target.value))} />

                  <label>Rotate Angle: {rotateAngle}°</label>
                  <input type="range" min="-180" max="180" step="1" value={rotateAngle} onChange={(e) => setRotateAngle(Number(e.target.value))} />

                  <div className="check-grid">
                    <label className="check-item"><input type="checkbox" checked={edgeDetection} onChange={(e) => setEdgeDetection(e.target.checked)} /><span>Edge Detection</span></label>
                    <label className="check-item"><input type="checkbox" checked={flipHorizontal} onChange={(e) => setFlipHorizontal(e.target.checked)} /><span>Flip Horizontal</span></label>
                    <label className="check-item"><input type="checkbox" checked={flipVertical} onChange={(e) => setFlipVertical(e.target.checked)} /><span>Flip Vertical</span></label>
                  </div>
                </>
              )}

              {activeTab === 'run' && (
                <>
                  <label>Image</label>
                  <input type="file" accept="image/png,image/jpeg" onChange={(e) => setFile(e.target.files?.[0] || null)} />

                  <div className="check-grid">
                    <label className="check-item"><input type="checkbox" checked={autoRun} onChange={(e) => setAutoRun(e.target.checked)} /><span>Auto-run Inference</span></label>
                  </div>

                  <button disabled={loading} onClick={runInference}>
                    {loading ? 'Processing...' : 'Run Inference Now'}
                  </button>

                  <div className="run-actions">
                    <button type="button" className="secondary-btn" onClick={() => inferenceCacheRef.current.clear()}>
                      Clear Cache
                    </button>
                    <button type="button" className="secondary-btn" onClick={resetAll}>
                      Reset Controls
                    </button>
                  </div>
                </>
              )}

              {error && <p className="error">{error}</p>}
            </div>
          </section>

          <section className="panel result">
            <h2>Detection Output</h2>

            {outputImage ? (
              <>
                {showPreprocessPreview && preprocessedImage && (
                  <div className="preview-block">
                    <h3>After Preprocessing</h3>
                    <img src={preprocessedImage} alt="Preprocessed" />
                  </div>
                )}

                <div className="preview-block">
                  <h3>Detection Result</h3>
                  <div className={`annotation-stage ${hitlEnabled ? 'active' : ''}`} onClick={onImageClick}>
                    <img ref={outputImgRef} src={outputImage} alt="Processed output" />
                    {hitlEnabled && outputImgRef.current && (
                      <svg
                        className="annotation-overlay"
                        viewBox={`0 0 ${outputImgRef.current.naturalWidth || 1} ${outputImgRef.current.naturalHeight || 1}`}
                        preserveAspectRatio="none"
                      >
                        {annotationPoints.length >= 2 && (
                          <polyline points={toPointString(annotationPoints)} className="annotation-polyline" />
                        )}
                        {annotationPoints.length >= 3 && (
                          <polygon points={toPointString(annotationPoints)} className="annotation-polygon" />
                        )}
                        {annotationPoints.map((p, idx) => (
                          <circle key={`${p.x}-${p.y}-${idx}`} cx={p.x} cy={p.y} r="4" className="annotation-point" />
                        ))}
                      </svg>
                    )}
                  </div>
                  <p className="hint">{hitlEnabled ? 'Click the image to place polygon points. Click anywhere on page to spawn network nodes.' : 'Enable Human-in-the-Loop (HITL) to begin manual correction.'}</p>
                </div>

                <div className="hitl-box">
                  <h3>Human-in-the-Loop (HITL) Re-annotation</h3>
                  <div className="check-grid">
                    <label className="check-item">
                      <input type="checkbox" checked={hitlEnabled} onChange={(e) => setHitlEnabled(e.target.checked)} />
                      <span>Enable Human-in-the-Loop (HITL) Re-annotation</span>
                    </label>
                  </div>

                  <label>Target Model</label>
                  <select value={annotationTarget} onChange={(e) => setAnnotationTarget(e.target.value)}>
                    {selectedModels.map((m) => (
                      <option key={m} value={m}>{m}</option>
                    ))}
                  </select>

                  <label>Notes</label>
                  <input value={annotationNotes} onChange={(e) => setAnnotationNotes(e.target.value)} placeholder="Optional notes" />

                  <div className="hitl-actions">
                    <button onClick={() => setAnnotationPoints((prev) => prev.slice(0, -1))}>Undo Point</button>
                    <button onClick={() => setAnnotationPoints([])}>Clear Polygon</button>
                    <button onClick={saveAnnotation}>Save Annotation</button>
                  </div>
                  {annotationMessage && <p className="hint">{annotationMessage}</p>}
                </div>

                <h3>Per-model Stats</h3>
                <div className="stats-grid">
                  {Object.entries(result.model_stats || {}).map(([model, stats]) => (
                    <div className="stat-card" key={model}>
                      <strong>{model}</strong>
                      <div>Threshold Mode: {stats.threshold_mode}</div>
                      <div>Threshold Used: {stats.threshold_used}</div>
                      <div>Coverage: {stats.coverage_pct}%</div>
                      <div>Positive Pixels: {stats.positive_pixels}</div>
                    </div>
                  ))}
                </div>

                <h3>Latest Saved Annotations</h3>
                <div className="stats-grid">
                  {savedAnnotations.length ? (
                    savedAnnotations.slice().reverse().slice(0, 6).map((item) => (
                      <div className="stat-card" key={item.id}>
                        <strong>{item.target_model}</strong>
                        <div>ID: {item.id.slice(0, 8)}...</div>
                        <div>Points: {item.points.length}</div>
                        <div>{new Date(item.created_at).toLocaleString()}</div>
                      </div>
                    ))
                  ) : (
                    <div className="stat-card"><div>No annotations yet.</div></div>
                  )}
                </div>
              </>
            ) : (
              <p className="placeholder">Upload an image and run inference.</p>
            )}
          </section>
        </main>
      </div>
    </>
  )
}

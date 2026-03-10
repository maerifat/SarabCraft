export default function DocsPage() {
  return (
    <div className="space-y-8 max-w-3xl">
      <div>
        <h3 className="text-sm font-semibold text-slate-200 mb-1">Plugin Documentation</h3>
        <p className="text-xs text-slate-500">Everything you need to build custom classifiers for SarabCraft.</p>
      </div>

      {/* Overview */}
      <Section title="Overview" id="overview">
        <P>
          Plugins let you extend the Transfer Verification step with your own classification logic.
          After an adversarial image or audio sample is generated, it can be tested against any number
          of plugins alongside the built-in models and cloud services.
        </P>
        <P>
          A plugin is a single Python <C>.py</C> file that exports a <C>classify()</C> function.
          The app calls that function with the adversarial sample (and optionally the original) and
          expects a list of predictions back. That's it &mdash; no framework, no server, no HTTP.
        </P>
      </Section>

      {/* Quick Start */}
      <Section title="Quick Start" id="quickstart">
        <P>Create a file called <C>my_classifier.py</C> with the following content:</P>
        <Code>{`PLUGIN_NAME = "My Classifier"
PLUGIN_TYPE = "image"                       # "image", "audio", or "both"
PLUGIN_DESCRIPTION = "A simple demo plugin" # optional, shown in the UI

def classify(adversarial_image, *, original_image=None, config={}):
    """
    adversarial_image : PIL.Image.Image  — the attacked image
    original_image    : PIL.Image.Image  — the clean image (may be None)
    config            : dict             — global variables from Settings > Variables

    Must return a list of dicts: [{"label": str, "confidence": float}, ...]
    """
    return [
        {"label": "cat",  "confidence": 0.85},
        {"label": "dog",  "confidence": 0.10},
        {"label": "bird", "confidence": 0.05},
    ]`}</Code>
        <P>
          Then go to <strong>Settings &rarr; Plugins &rarr; Add Plugin</strong> and either upload the
          file or paste the code in the inline editor.
        </P>
      </Section>

      {/* Plugin Contract */}
      <Section title="Plugin Contract" id="contract">
        <SubHead>Required exports</SubHead>
        <Table rows={[
          ['PLUGIN_NAME', 'str', 'Yes', 'Display name shown in the UI.'],
          ['PLUGIN_TYPE', '"image" | "audio" | "both"', 'Yes', 'Determines which data your plugin receives.'],
          ['PLUGIN_DESCRIPTION', 'str', 'No', 'Short description, shown on the plugin card.'],
          ['classify()', 'function', 'Yes', 'The function the app calls. Signature depends on PLUGIN_TYPE.'],
        ]} />

        <SubHead>classify() signatures</SubHead>

        <div className="space-y-4">
          <div>
            <Badge color="blue">Image Plugin</Badge>
            <Code>{`def classify(adversarial_image, *, original_image=None, config={}):
    # adversarial_image: PIL.Image.Image (RGB)
    # original_image:    PIL.Image.Image or None
    # config:            dict of global variables
    return [{"label": "cat", "confidence": 0.95}]`}</Code>
          </div>

          <div>
            <Badge color="purple">Audio Plugin</Badge>
            <Code>{`def classify(adversarial_audio, *, original_audio=None, sample_rate=16000, config={}):
    # adversarial_audio: numpy.ndarray (1-D float32 waveform)
    # original_audio:    numpy.ndarray or None
    # sample_rate:       int (e.g. 16000)
    # config:            dict of global variables
    return [{"label": "yes", "confidence": 0.92}]`}</Code>
          </div>
        </div>

        <SubHead>Return format</SubHead>
        <P>
          <C>classify()</C> must return a <C>list</C> of dicts. Each dict needs two keys:
        </P>
        <Table rows={[
          ['label', 'str', 'Yes', 'The predicted class name.'],
          ['confidence', 'float', 'Yes', 'Confidence score between 0.0 and 1.0.'],
        ]} />
        <P>
          Return results sorted by confidence (highest first). Only the top 5 are displayed in the UI.
        </P>
      </Section>

      {/* Using Variables & Secrets */}
      <Section title="Using Variables & Secrets" id="variables">
        <P>
          Plugins often need API keys, tokens, or other configuration values. Instead of hard-coding
          them into your plugin code, use the <strong>global variables</strong> system under
          <strong> Settings &rarr; Variables</strong>.
        </P>

        <SubHead>How it works</SubHead>
        <ol className="list-decimal list-inside space-y-2 text-xs text-slate-400">
          <li>Go to <strong className="text-slate-300">Settings &rarr; Variables</strong> and add a variable (e.g. key = <C>OPENAI_API_KEY</C>, value = <C>sk-...</C>). Mark it as <strong className="text-slate-300">Masked</strong> to hide the value in the UI.</li>
          <li>In your plugin, read it from the <C>config</C> dict that is automatically passed to <C>classify()</C>.</li>
          <li>The app injects <em>all</em> global variables into <C>config</C> at runtime &mdash; always unmasked.</li>
        </ol>

        <SubHead>Example: Calling OpenAI Vision API</SubHead>
        <Code>{`import base64, io, json, urllib.request

PLUGIN_NAME = "OpenAI Vision"
PLUGIN_TYPE = "image"
PLUGIN_DESCRIPTION = "Classifies images via OpenAI GPT-4 Vision"

def classify(adversarial_image, *, original_image=None, config={}):
    api_key = config.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY in Settings > Variables")

    # Encode image to base64
    buf = io.BytesIO()
    adversarial_image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    body = json.dumps({
        "model": "gpt-4o",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Classify this image. Return only the object name."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        "max_tokens": 50,
    }).encode()

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())

    label = result["choices"][0]["message"]["content"].strip()
    return [{"label": label, "confidence": 1.0}]`}</Code>

        <SubHead>Example: Calling a Custom REST API</SubHead>
        <Code>{`import io, json, urllib.request

PLUGIN_NAME = "My Company API"
PLUGIN_TYPE = "image"
PLUGIN_DESCRIPTION = "Sends images to our internal classifier"

def classify(adversarial_image, *, original_image=None, config={}):
    api_url  = config.get("MY_API_URL", "https://api.example.com/classify")
    api_token = config.get("MY_API_TOKEN", "")

    buf = io.BytesIO()
    adversarial_image.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    # Build multipart request
    import uuid
    boundary = uuid.uuid4().hex
    body  = f"--{boundary}\\r\\n".encode()
    body += b'Content-Disposition: form-data; name="image"; filename="img.png"\\r\\n'
    body += b"Content-Type: image/png\\r\\n\\r\\n"
    body += image_bytes + b"\\r\\n"
    body += f"--{boundary}--\\r\\n".encode()

    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    req = urllib.request.Request(api_url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())

    # Adapt the response to the expected format
    return [
        {"label": p["class"], "confidence": p["score"]}
        for p in result.get("predictions", [])
    ]`}</Code>

        <SubHead>Example: Using AWS Rekognition</SubHead>
        <Code>{`import io, json

PLUGIN_NAME = "AWS Rekognition"
PLUGIN_TYPE = "image"
PLUGIN_DESCRIPTION = "Classifies via AWS Rekognition DetectLabels"

def classify(adversarial_image, *, original_image=None, config={}):
    import boto3

    session = boto3.Session(
        aws_access_key_id=config.get("AWS_ACCESS_KEY_ID", ""),
        aws_secret_access_key=config.get("AWS_SECRET_ACCESS_KEY", ""),
        region_name=config.get("AWS_REGION", "us-east-1"),
    )
    client = session.client("rekognition")

    buf = io.BytesIO()
    adversarial_image.save(buf, format="PNG")

    resp = client.detect_labels(
        Image={"Bytes": buf.getvalue()},
        MaxLabels=5,
    )
    return [
        {"label": lbl["Name"], "confidence": lbl["Confidence"] / 100.0}
        for lbl in resp["Labels"]
    ]`}</Code>

        <SubHead>Example: Using HuggingFace Inference API</SubHead>
        <Code>{`import io, json, urllib.request

PLUGIN_NAME = "HuggingFace ViT"
PLUGIN_TYPE = "image"
PLUGIN_DESCRIPTION = "Classifies via HuggingFace Inference API"

def classify(adversarial_image, *, original_image=None, config={}):
    hf_token = config.get("HF_TOKEN", "")
    model_id = config.get("HF_MODEL_ID", "google/vit-base-patch16-224")

    buf = io.BytesIO()
    adversarial_image.save(buf, format="PNG")

    url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/octet-stream",
    }

    req = urllib.request.Request(url, data=buf.getvalue(), headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        results = json.loads(resp.read())

    return [
        {"label": r["label"], "confidence": r["score"]}
        for r in results[:5]
    ]`}</Code>
      </Section>

      {/* Installation Methods */}
      <Section title="Installation Methods" id="install">
        <SubHead>1. Upload a .py file</SubHead>
        <P>
          Click <strong>Add Plugin &rarr; Upload .py / .zip</strong>. Drag and drop or browse for a file.
          The app validates that it exports <C>PLUGIN_NAME</C> and <C>classify()</C> before saving.
        </P>

        <SubHead>2. Upload a .zip archive</SubHead>
        <P>
          If your plugin spans multiple files, zip them together. The app extracts all <C>.py</C> files
          and installs each one that passes validation. Files starting with <C>_</C> are skipped.
        </P>

        <SubHead>3. Write code inline</SubHead>
        <P>
          Click <strong>Add Plugin &rarr; Write Code</strong> to open the inline editor. This is great
          for quick prototyping &mdash; just paste your code and hit Save.
        </P>
      </Section>

      {/* Testing */}
      <Section title="Testing Plugins" id="testing">
        <P>
          Each plugin card has a three-dot menu with a <strong>Test</strong> option. This sends a dummy
          224&times;224 gray image (or silent audio sample) through your <C>classify()</C> function and
          reports whether it returned valid predictions.
        </P>
        <P>
          In production, your plugin runs during <strong>Transfer Verification</strong>. Select it in the
          verification panel alongside built-in services, and its predictions appear in the results table.
        </P>
      </Section>

      {/* Available Libraries */}
      <Section title="Available Libraries" id="libraries">
        <P>
          Plugins run in the same Python environment as the app. The following are pre-installed and ready
          to import:
        </P>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 mt-2">
          {['numpy', 'PIL (Pillow)', 'torch', 'torchvision', 'torchaudio', 'transformers',
            'scipy', 'soundfile', 'boto3', 'requests*', 'urllib (stdlib)'].map(lib => (
            <span key={lib} className="text-[11px] px-2.5 py-1.5 rounded-lg bg-slate-800/60 text-slate-300 border border-slate-700/40 font-mono">
              {lib}
            </span>
          ))}
        </div>
        <P className="mt-3">
          <em className="text-slate-600">* If a library is not available, use <C>urllib.request</C> from the standard library.</em>
        </P>
      </Section>

      {/* Error Handling */}
      <Section title="Error Handling" id="errors">
        <P>
          If your <C>classify()</C> function raises an exception, the plugin card will show an error
          indicator and the traceback is available in the API response. Common issues:
        </P>
        <ul className="list-disc list-inside space-y-1.5 text-xs text-slate-400 mt-2">
          <li><strong className="text-slate-300">Missing PLUGIN_NAME</strong> &mdash; Your file must define a top-level <C>PLUGIN_NAME = "..."</C> string.</li>
          <li><strong className="text-slate-300">Missing classify()</strong> &mdash; Must export a function called <C>classify</C>.</li>
          <li><strong className="text-slate-300">Wrong return format</strong> &mdash; Must return <C>[{`{"label": str, "confidence": float}`}]</C>.</li>
          <li><strong className="text-slate-300">Missing config variable</strong> &mdash; Check that the variable is set in <strong>Settings &rarr; Variables</strong>.</li>
          <li><strong className="text-slate-300">Import error</strong> &mdash; The library may not be installed in the app's environment.</li>
        </ul>
      </Section>

      {/* Best Practices */}
      <Section title="Best Practices" id="best-practices">
        <ul className="list-disc list-inside space-y-2 text-xs text-slate-400">
          <li><strong className="text-slate-300">Never hard-code secrets.</strong> Use <C>config.get("KEY")</C> and store values in Settings &rarr; Variables.</li>
          <li><strong className="text-slate-300">Handle timeouts.</strong> Remote API calls should include a <C>timeout</C> parameter.</li>
          <li><strong className="text-slate-300">Return sorted predictions.</strong> Highest confidence first. The UI displays the top 5.</li>
          <li><strong className="text-slate-300">Keep plugins focused.</strong> One classifier per file. If you need shared utilities, import from the standard library.</li>
          <li><strong className="text-slate-300">Test locally first.</strong> Use the Test button to verify your plugin loads and returns predictions in the expected format.</li>
          <li><strong className="text-slate-300">Add a PLUGIN_DESCRIPTION.</strong> It shows on the plugin card and helps other users understand what it does.</li>
        </ul>
      </Section>
    </div>
  )
}

function Section({ title, id, children }) {
  return (
    <section id={id} className="space-y-3">
      <h4 className="text-xs font-semibold text-slate-200 uppercase tracking-wider border-b border-slate-700/50 pb-2">{title}</h4>
      {children}
    </section>
  )
}

function SubHead({ children }) {
  return <h5 className="text-xs font-medium text-slate-300 mt-4 mb-1">{children}</h5>
}

function P({ children, className = '' }) {
  return <p className={`text-xs text-slate-400 leading-relaxed ${className}`}>{children}</p>
}

function C({ children }) {
  return <code className="text-[11px] bg-slate-800/80 text-[var(--accent)] px-1.5 py-0.5 rounded font-mono">{children}</code>
}

function Badge({ children, color = 'blue' }) {
  const colors = {
    blue: 'bg-blue-500/15 text-blue-400 border-blue-500/20',
    purple: 'bg-purple-500/15 text-purple-400 border-purple-500/20',
  }
  return (
    <span className={`inline-block text-[9px] px-2 py-0.5 rounded border font-semibold uppercase tracking-wide mb-2 ${colors[color]}`}>
      {children}
    </span>
  )
}

function Code({ children }) {
  return (
    <pre className="bg-slate-950 border border-slate-800 rounded-lg p-4 text-[11px] text-slate-300 font-mono overflow-x-auto whitespace-pre leading-relaxed">
      {children}
    </pre>
  )
}

function Table({ rows }) {
  return (
    <div className="overflow-x-auto mt-2">
      <table className="w-full text-[11px]">
        <thead>
          <tr className="text-left text-slate-500 border-b border-slate-700/50">
            <th className="py-1.5 pr-3 font-medium">Name</th>
            <th className="py-1.5 pr-3 font-medium">Type</th>
            <th className="py-1.5 pr-3 font-medium">Required</th>
            <th className="py-1.5 font-medium">Description</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([name, type, required, desc], i) => (
            <tr key={i} className="border-b border-slate-800/50">
              <td className="py-1.5 pr-3 font-mono text-[var(--accent)]">{name}</td>
              <td className="py-1.5 pr-3 text-slate-500 font-mono">{type}</td>
              <td className="py-1.5 pr-3 text-slate-500">{required}</td>
              <td className="py-1.5 text-slate-400">{desc}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

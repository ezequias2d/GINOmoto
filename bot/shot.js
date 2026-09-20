// Screenshot service: keeps a headless Chrome pointed at the prismarine-viewer page (the bot's first-person view of the
// world) and exports frames as PNG, so the jev vision model gets real pixels instead of only the symbolic world state.
//
// Software WebGL (SwiftShader) renders that page on the CPU, which is several cores of heat on a laptop — so the page is
// *frozen* between shots (Page.setWebLifecycleState) and woken up only when a frame is requested.
//
//   node bot/shot.js --viewer 3007 --port 3008 --w 480 --h 270
//   POST /shot  {"out":"/abs/path.png"}  -> {ok, file, ms, bytes}
//   GET  /shot.png                       -> the last frame
//   GET  /freeze | /wake                 -> park / unpark the page by hand
const puppeteer = require('puppeteer-core')
const http = require('http')
const fs = require('fs')
const path = require('path')

const argv = process.argv.slice(2)
const opt = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d }
const VIEWER = +opt('--viewer', 3007), PORT = +opt('--port', 3008)
const W = +opt('--w', 480), H = +opt('--h', 270)
const WAKE = +opt('--wake', 2000)

let page, cdp, last = Buffer.alloc(0), ready = false
const sleep = ms => new Promise(r => setTimeout(r, ms))

;(async () => {
  const browser = await puppeteer.launch({
    executablePath: process.env.CHROME || '/usr/bin/google-chrome',
    headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--enable-unsafe-swiftshader',
           '--use-gl=angle', '--use-angle=swiftshader', `--window-size=${W},${H}`]
  })
  page = await browser.newPage()
  await page.setViewport({ width: W, height: H })
  cdp = await page.target().createCDPSession()
  const url = `http://127.0.0.1:${VIEWER}/`
  for (let attempt = 0; attempt < 60; attempt++) {
    try { await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 15000 }); break } catch (e) { await sleep(1000) }
  }
  await page.waitForSelector('canvas', { timeout: 60000 })
  await sleep(8000)          // let the viewer stream the chunks around the bot and draw a first real frame
  last = await page.screenshot({ type: 'png' })
  ready = true
  console.log('shot service on', PORT, 'viewer', url, `${W}x${H}`, `warm frame ${last.length} bytes`)
  await freeze()             // idle near 0% cpu until the next frame is asked for
})().catch(e => { console.log('shot service failed:', e.message); process.exit(1) })

const setState = async s => { try { await cdp.send('Page.setWebLifecycleState', { state: s }) } catch (e) {} }
const freeze = () => setState('frozen')
const wake = () => setState('active')

async function shot() {
  const t0 = Date.now()
  await wake()
  await sleep(WAKE)          // one or two rendered frames: the socket kept streaming while frozen
  last = await page.screenshot({ type: 'png' })
  const wakeMs = Date.now() - t0
  await freeze()
  return { buf: last, wakeMs }
}

http.createServer((req, res) => {
  let body = ''
  req.on('data', c => { body += c })
  req.on('end', async () => {
    if (req.url === '/shot.png') {
      if (!last.length) return res.writeHead(503).end()
      res.writeHead(200, { 'content-type': 'image/png', 'cache-control': 'no-store' }); return res.end(last)
    }
    if (req.url === '/freeze') { await freeze(); res.writeHead(200).end('{}'); return }
    if (req.url === '/wake') { await wake(); res.writeHead(200).end('{}'); return }
    if (!ready || req.url !== '/shot') { res.writeHead(404); return res.end('{}') }
    try {
      const a = JSON.parse(body || '{}')
      const { buf, wakeMs } = await shot()
      const out = path.resolve(a.out || path.join(__dirname, '..', 'shots', 'latest.png'))
      fs.mkdirSync(path.dirname(out), { recursive: true })
      fs.writeFileSync(out, buf)
      res.writeHead(200, { 'content-type': 'application/json' }); res.end(JSON.stringify({ ok: true, file: out, ms: wakeMs, bytes: buf.length }))
    } catch (e) {
      res.writeHead(500, { 'content-type': 'application/json' }); res.end(JSON.stringify({ ok: false, msg: e.message }))
    }
  })
}).listen(PORT, '127.0.0.1')

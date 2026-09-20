// Mineflayer skill server for the jev agent (jev/agent.py): the NLI model picks a skill, this process executes it in a real
// Minecraft world. Adapted from AlexWortega/openjev code/mc_bot.js (MIT) — ports and the `say` skill changed.
//   node skill_server.js [--port 25565] [--http 3010] [--viewer 3007]
//   GET  /state                      -> inventory / stations / visible blocks
//   POST /act   {skill, arg, n}      -> {ok, msg}   skills: collect | craft | place | smelt | explore | say
//   POST /reset {x, z}               -> clear inventory, teleport to a fresh spot
const mineflayer = require('mineflayer')
const { pathfinder, Movements, goals } = require('mineflayer-pathfinder')
const { Vec3 } = require('vec3')
const http = require('http')

const argv = process.argv.slice(2)
const opt = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d }
const bot = mineflayer.createBot({ host: opt('--host', '127.0.0.1'), port: +opt('--port', 25565),
                                   username: opt('--name', 'GINOmoto'), version: opt('--version', '1.21.4'),
                                   auth: opt('--auth', 'offline') })
// `--op` means the bot is server operator (own test server): only then are gamerules/teleports issued, they are
// pointless (and noisy in chat) on somebody else's world.
const IS_OP = argv.includes('--op')
bot.loadPlugin(pathfinder)
let mcData, ready = false
const sleep = ms => new Promise(r => setTimeout(r, ms))

// text-level groups: the planner only ever sees these names
const BLOCKS = { log: n => n.endsWith('_log') && !n.startsWith('stripped'), stone: n => n === 'stone', iron_ore: n => n === 'iron_ore' || n === 'deepslate_iron_ore' }
const ITEMS = { log: n => n.endsWith('_log'), planks: n => n.endsWith('_planks'), stick: n => n === 'stick', cobblestone: n => n === 'cobblestone' }
const DROPS = { log: 'log', stone: 'cobblestone', iron_ore: 'raw_iron' }
const RADIUS = { log: 48, stone: 32, iron_ore: 64 }
const itemMatch = g => ITEMS[g] || (n => n === g)
const count = g => bot.inventory.items().filter(i => itemMatch(g)(i.name)).reduce((s, i) => s + i.count, 0)
const blockIds = g => Object.values(mcData.blocksByName).filter(b => BLOCKS[g](b.name)).map(b => b.id)
const bad = new Set()

// --- chat: the player talks to the bot in game, the jev agent reads this buffer ------------------------------------
const chatLog = []
bot.on('chat', (username, message) => {
  if (username === bot.username) return
  chatLog.push({ username, message, t: Date.now() })
  if (chatLog.length > 300) chatLog.shift()
})
bot.on('whisper', (username, message) => {
  chatLog.push({ username, message, t: Date.now(), whisper: true })
  if (chatLog.length > 300) chatLog.shift()
})
const otherPlayers = () => Object.values(bot.players).filter(p => p.username !== bot.username && p.entity)
const anyPlayer = () => otherPlayers()[0]

bot.once('spawn', async () => {
  mcData = require('minecraft-data')(bot.version)
  const mv = new Movements(bot)
  mv.allow1by1towers = true; mv.canDig = true; mv.allowParkour = true
  bot.pathfinder.setMovements(mv)
  bot.pathfinder.thinkTimeout = 10000
  if (IS_OP) for (const c of ['gamerule doDaylightCycle false', 'time set day', 'gamerule doWeatherCycle false', 'weather clear', 'gamerule keepInventory true', 'difficulty peaceful']) { bot.chat('/' + c); await sleep(150) }
  if (opt('--viewer', null)) require('prismarine-viewer').mineflayer(bot, { port: +opt('--viewer'), firstPerson: opt('--view', 'first') === 'first', viewDistance: 5 })
  ready = true
  console.log('bot ready at', bot.entity.position.floored().toString())
})
bot.on('kicked', r => console.log('kicked', r))
bot.on('error', e => console.log('error', e.message))
bot.on('death', () => console.log('bot died'))

function findNear(g) {
  const pos = bot.findBlocks({ matching: blockIds(g), maxDistance: RADIUS[g], count: 64 }).filter(p => !bad.has(p.toString()))
  const me = bot.entity.position
  // prefer exposed blocks that are not high above the head (tree tops need towers, buried blocks need tunnels)
  const exposed = p => [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]].some(d => { const b = bot.blockAt(p.offset(...d)); return b && b.name === 'air' })
  const cost = p => p.distanceTo(me) + 4 * Math.max(0, p.y - me.y - 3) + (p.distanceTo(me) < 24 && !exposed(p) ? 10 : 0)
  return pos.map(p => [cost(p), p]).sort((a, b) => a[0] - b[0]).map(x => x[1])
}
const station = name => bot.findBlock({ matching: mcData.blocksByName[name].id, maxDistance: 32 })

function state() {
  const inv = {}
  for (const i of bot.inventory.items()) {
    const g = Object.keys(ITEMS).find(k => ITEMS[k](i.name)) || i.name
    inv[g] = (inv[g] || 0) + i.count
  }
  const p = bot.entity.position
  return {
    inventory: inv, pos: [Math.round(p.x), Math.round(p.y), Math.round(p.z)], health: bot.health,
    near: { crafting_table: !!station('crafting_table'), furnace: !!station('furnace') },
    visible: Object.fromEntries(Object.keys(BLOCKS).map(g => [g, findNear(g).length > 0])),
    players: Object.fromEntries(otherPlayers().map(p => [p.username, p.entity.position.floored().toArray()])),
    bot: bot.username
  }
}

async function withTimeout(p, ms, what) {
  let t
  // stop() with no active path would cancel the *next* goto, so only clean up after a real timeout
  const timer = new Promise((_, rej) => { t = setTimeout(() => { bot.pathfinder.setGoal(null); try { bot.stopDigging() } catch (e) {} ; rej(new Error(`timeout: ${what}`)) }, ms) })
  try { return await Promise.race([p, timer]) } finally { clearTimeout(t) }
}

async function pickupNear(pos) {
  await sleep(350)
  for (let k = 0; k < 3; k++) {
    const e = Object.values(bot.entities).filter(e => e.name === 'item' && e.position.distanceTo(pos) < 6).sort((a, b) => a.position.distanceTo(bot.entity.position) - b.position.distanceTo(bot.entity.position))[0]
    if (!e) return
    try { await withTimeout(bot.pathfinder.goto(new goals.GoalNear(e.position.x, e.position.y, e.position.z, 0.7)), 15000, 'pickup') } catch (err) { return }
    await sleep(300)
  }
}

async function collect(g, n) {
  if (!BLOCKS[g]) return { ok: false, msg: `unknown block group ${g}` }
  const drop = DROPS[g], start = count(drop)
  for (let tries = 0; count(drop) < start + n && tries < n * 3 + 4; tries++) {
    const cands = findNear(g)
    if (!cands.length) break
    const block = bot.blockAt(cands[0])
    try {
      await withTimeout(bot.pathfinder.goto(new goals.GoalNear(block.position.x, block.position.y, block.position.z, 3)), 60000, 'path to ' + g)
      if (!bot.canDigBlock(block)) throw new Error('block out of reach')
      const tool = bot.pathfinder.bestHarvestTool(block)
      if (tool) await bot.equip(tool, 'hand')
      if (!block.canHarvest(bot.heldItem ? bot.heldItem.type : null)) return { ok: false, msg: `cannot harvest ${g} without a suitable pickaxe` }
      await withTimeout(bot.dig(block), 30000, 'dig ' + g)
      await pickupNear(block.position)
    } catch (e) { bad.add(cands[0].toString()); console.log('collect err', e.message) }
  }
  const got = count(drop) - start
  return got > 0 ? { ok: true, msg: `collected ${got} ${drop}` } : { ok: false, msg: `no reachable ${g} block found nearby` }
}

// bot.craft() clicks the next ingredient while the previous one is still on the cursor (swap -> garbage in the grid),
// so the clicks are done here: pick a stack, right-click one item into each grid slot, put the rest back, take the result.
async function craftOnce(recipe, table) {
  const win = recipe.requiresTable ? await bot.openBlock(table) : bot.inventory, w = recipe.requiresTable ? 3 : 2
  const click = async (slot, btn) => { await bot.clickWindow(slot, btn, 0); await sleep(60) }
  try {
    const dests = {}
    if (recipe.inShape) recipe.inShape.forEach((row, y) => row.forEach((ing, x) => { if (ing.id !== -1) (dests[ing.id] = dests[ing.id] || []).push(1 + x + w * y) }))
    else recipe.ingredients.forEach((ing, k) => (dests[ing.id] = dests[ing.id] || []).push(1 + k))
    for (const [id, slots] of Object.entries(dests)) {
      let from = null
      for (const d of slots) {
        if (!win.selectedItem) {  // stacks may be fragmented (4 + 4 + 2 planks): take the next one when the cursor runs empty
          const src = win.findInventoryItem(+id, null)
          if (!src) throw new Error('missing ingredient')
          from = src.slot
          await click(from, 0)
        }
        await click(d, 1)
      }
      if (win.selectedItem) await click(from, 0)
    }
    for (let i = 0; i < 40 && !win.slots[0]; i++) await sleep(50)
    if (!win.slots[0]) throw new Error('no crafting result appeared')
    await click(0, 0)
    await click(win.firstEmptyInventorySlot(), 0)
  } finally {
    if (recipe.requiresTable) bot.closeWindow(win)
    await sleep(250)
  }
}

async function craft(name) {
  const ids = Object.values(mcData.itemsByName).filter(i => itemMatch(name)(i.name)).map(i => i.id)
  const table = station('crafting_table')
  let recipe = null
  for (const id of ids) { recipe = bot.recipesFor(id, null, 1, table || null)[0]; if (recipe) break }
  if (!recipe) return { ok: false, msg: `cannot craft ${name}: missing ingredients${table ? '' : ' or no crafting table nearby'}` }
  if (recipe.requiresTable) await withTimeout(bot.pathfinder.goto(new goals.GoalNear(table.position.x, table.position.y, table.position.z, 2)), 60000, 'path to table')
  const before = count(name)
  try { await craftOnce(recipe, table) } catch (e) { console.log('craft err', e.message); return { ok: false, msg: `crafting ${name} failed: ${e.message}` } }
  return { ok: count(name) > before, msg: `crafted ${count(name) - before} ${name}` }
}

async function place(name) {
  const item = bot.inventory.items().find(i => i.name === name)
  if (!item) return { ok: false, msg: `no ${name} in the inventory` }
  const p = bot.entity.position.floored()
  for (let r = 1; r <= 3; r++) for (let dx = -r; dx <= r; dx++) for (let dz = -r; dz <= r; dz++) for (const dy of [0, 1, -1]) {
    if (Math.max(Math.abs(dx), Math.abs(dz)) !== r) continue
    const ref = bot.blockAt(p.offset(dx, dy - 1, dz)), above = bot.blockAt(p.offset(dx, dy, dz))
    if (!ref || !above || ref.boundingBox !== 'block' || above.name !== 'air') continue
    try {
      await bot.equip(item, 'hand')
      await bot.lookAt(ref.position.offset(0.5, 1, 0.5))
      await bot.placeBlock(ref, new Vec3(0, 1, 0))
      return { ok: true, msg: `placed ${name}` }
    } catch (e) { console.log('place err', e.message) }
  }
  return { ok: false, msg: `no free spot to place ${name}` }
}

async function smelt(input) {
  const fb = station('furnace')
  if (!fb) return { ok: false, msg: 'no furnace nearby' }
  const raw = bot.inventory.items().find(i => i.name === input), fuel = bot.inventory.items().find(i => ITEMS.planks(i.name) || i.name === 'coal')
  if (!raw) return { ok: false, msg: `no ${input} in the inventory` }
  if (!fuel) return { ok: false, msg: 'no fuel (planks) in the inventory' }
  await withTimeout(bot.pathfinder.goto(new goals.GoalNear(fb.position.x, fb.position.y, fb.position.z, 2)), 60000, 'path to furnace')
  const f = await bot.openFurnace(fb), n = raw.count
  const nfuel = Math.min(fuel.count, fuel.name === 'coal' ? Math.ceil(n / 8) : Math.ceil(n * 2 / 3))
  await f.putFuel(fuel.type, null, nfuel); await f.putInput(raw.type, null, n)
  for (let t = 0; t < n * 12 + 8; t++) { await sleep(1000); const o = f.outputItem(); if (o && o.count >= n) break; if (!f.inputItem() && f.fuel <= 0 && t > 3) break }
  const out = f.outputItem()
  if (out) await f.takeOutput()
  if (f.inputItem()) await f.takeInput()
  f.close()
  return out ? { ok: true, msg: `smelted ${out.count} ${out.name}` } : { ok: false, msg: 'smelting produced nothing (not enough fuel?)' }
}

let heading = Math.random() * 2 * Math.PI  // kept per episode: a fresh random direction every call is a random walk that goes nowhere
async function explore(arg) {
  if (typeof arg === 'number') heading = (arg * Math.PI) / 180  // the vision layer can steer us, in degrees
  const a = heading + (Math.random() - 0.5) * 0.6, p = bot.entity.position.clone()
  try { await withTimeout(bot.pathfinder.goto(new goals.GoalXZ(p.x + 60 * Math.cos(a), p.z + 60 * Math.sin(a))), 60000, 'explore') } catch (e) { heading += 1.2 }
  return { ok: true, msg: `walked ${Math.round(bot.entity.position.distanceTo(p))} blocks` }
}

// --- skills the chat layer drives ---------------------------------------------------------------------------------
const pname = a => (typeof a === 'string' && a.trim()) ? a.trim().replace(/^@/, '') : null

function findPlayer(name) {
  const ps = otherPlayers()
  return name ? ps.find(p => p.username.toLowerCase() === name.toLowerCase()) : ps[0]
}

async function follow(name) {
  const p = findPlayer(pname(name))
  if (!p) return { ok: false, msg: 'no player in sight to follow' }
  bot.pathfinder.setGoal(new goals.GoalFollow(p.entity, 3), true)
  return { ok: true, msg: `following ${p.username}` }
}

async function come(name) {
  const p = findPlayer(pname(name))
  if (!p) return { ok: false, msg: 'no player in sight' }
  const pos = p.entity.position
  await withTimeout(bot.pathfinder.goto(new goals.GoalNear(pos.x, pos.y, pos.z, 2)), 60000, 'come')
  return { ok: true, msg: `walked to ${p.username}` }
}

async function goto(arg, y, z) {
  let tx, ty, tz
  if (typeof arg === 'string') {
    const p = arg.trim().split(/[\s,]+/).map(Number).filter(n => Number.isFinite(n))
    if (p.length === 2) [tx, tz] = p
    else if (p.length >= 3) [tx, ty, tz] = p
  } else if (Number.isFinite(+arg)) {
    tx = +arg; ty = Number.isFinite(+y) ? +y : undefined; tz = Number.isFinite(+z) ? +z : undefined
  }
  if (!Number.isFinite(tx) || !Number.isFinite(tz)) return { ok: false, msg: 'goto needs coordinates, e.g. "120 -400" or "120 70 -400"' }
  const goal = Number.isFinite(ty) ? new goals.GoalBlock(tx, ty, tz) : new goals.GoalXZ(tx, tz)
  await withTimeout(bot.pathfinder.goto(goal), 120000, 'goto')
  return { ok: true, msg: `walked to ${Math.round(tx)} ${Math.round(ty ?? 0)} ${Math.round(tz)}` }
}

async function stop() {
  try { bot.pathfinder.setGoal(null) } catch (e) {}
  try { bot.stopDigging() } catch (e) {}
  try { bot.clearControlStates() } catch (e) {}
  try { bot.deactivateItem() } catch (e) {}
  for (const e of Object.values(bot.entities)) { if (e.name && e !== bot.entity) break }
  return { ok: true, msg: 'stopped' }
}

function nearestEntity(match, maxDistance = 24) {
  return Object.values(bot.entities)
    .filter(e => e !== bot.entity && e.position.distanceTo(bot.entity.position) < maxDistance && match(e))
    .sort((a, b) => a.position.distanceTo(bot.entity.position) - b.position.distanceTo(bot.entity.position))[0]
}

async function attack(name) {
  const mobs = /zombie|skeleton|spider|creeper|witch|pillager|slime|creeper|enderman|hoglin|drowned|husk|zombified/i
  const target = nearestEntity(e => e.type === 'mob' && (name && name !== 'mob' ? new RegExp(name, 'i').test(e.name || '') : mobs.test(e.name || '')))
  if (!target) return { ok: false, msg: 'no hostile mob nearby' }
  const weapon = bot.inventory.items().find(i => /sword|axe/.test(i.name))
  if (weapon) await bot.equip(weapon, 'hand').catch(() => {})
  for (let i = 0; i < 60; i++) {
    const e = bot.entities[target.id]
    if (!e) break
    if (bot.entity.position.distanceTo(e.position) > 3) {
      try { await withTimeout(bot.pathfinder.goto(new goals.GoalNear(e.position.x, e.position.y, e.position.z, 2)), 8000, 'chase') } catch (err) {}
    }
    bot.lookAt(e.position.offset(0, 1, 0), true)
    bot.attack(e)
    await sleep(500)
  }
  return { ok: true, msg: `fought ${target.name}` }
}

function look(name) {
  const p = findPlayer(pname(name))
  if (!p) return { ok: false, msg: 'no player in sight' }
  bot.lookAt(p.entity.position.offset(0, 1.6, 0), true)
  return { ok: true, msg: `looking at ${p.username}` }
}

async function tower(n) {
  n = Math.min(Math.max(parseInt(n) || 3, 1), 10)
  const block = bot.inventory.items().find(i => i.name !== 'crafting_table' && bot.blockAt && /_planks|cobblestone|dirt|stone|log|sand/.test(i.name))
  if (!block) return { ok: false, msg: 'nothing to build with' }
  await bot.equip(block, 'hand')
  const y0 = bot.entity.position.y
  for (let i = 0; i < n * 4 && bot.entity.position.y < y0 + n; i++) {
    const below = bot.blockAt(bot.entity.position.offset(0, -1, 0))
    if (!below) break
    bot.setControlState('jump', true)
    await sleep(180)
    try { await bot.placeBlock(below, new Vec3(0, 1, 0)) } catch (e) {}
    await sleep(120)
  }
  bot.setControlState('jump', false)
  return { ok: true, msg: `tower: climbed ${Math.round(bot.entity.position.y - y0)} blocks` }
}

async function digdown(n) {
  n = Math.min(Math.max(parseInt(n) || 3, 1), 20)
  const start = bot.entity.position.clone()
  for (let i = 0; i < n; i++) {
    const b = bot.blockAt(bot.entity.position.offset(0, -1, 0))
    if (!b || b.boundingBox !== 'block') break
    try {
      const tool = bot.pathfinder.bestHarvestTool(b)
      if (tool) await bot.equip(tool, 'hand').catch(() => {})
      await bot.dig(b, true)
    } catch (e) { break }
    await sleep(150)
  }
  return { ok: true, msg: `dug ${Math.round(start.y - bot.entity.position.y)} blocks down` }
}

async function toss(name, n) {
  if (!name) return { ok: false, msg: 'what should I drop?' }
  const item = bot.inventory.items().find(i => itemMatch(name)(i.name))
  if (!item) return { ok: false, msg: `no ${name} in the inventory` }
  const count = Math.min(parseInt(n) || item.count, item.count)
  await bot.toss(item.type, null, count)
  return { ok: true, msg: `dropped ${count} ${item.name}` }
}

async function say(text) {
  if (!text || typeof text !== 'string') return { ok: false, msg: 'nothing to say' }
  bot.chat(text.slice(0, 250))
  return { ok: true, msg: `said: ${text}` }
}

async function reset(x, z) {
  if (!IS_OP) return { ok: false, msg: 'reset needs server operator (/clear, /spreadplayers)' }
  bot.chat('/clear'); await sleep(200)
  // start on dry land with trees in sight (otherwise half of the episodes begin in the ocean)
  for (let k = 0; k < 12; k++) {
    bot.chat(`/spreadplayers ${x + 300 * k} ${z} 0 40 false ${bot.username}`); await sleep(4000)
    const feet = bot.blockAt(bot.entity.position.floored())
    if (findNear('log').length > 3 && feet && feet.name !== 'water') break
  }
  bad.clear(); heading = Math.random() * 2 * Math.PI
  return { ok: true, msg: 'reset to ' + bot.entity.position.floored().toString() }
}

let busy = false
http.createServer((req, res) => {
  let body = ''
  req.on('data', c => { body += c })
  req.on('end', async () => {
    const send = o => { res.writeHead(200, { 'content-type': 'application/json' }); res.end(JSON.stringify(o)) }
    if (!ready) return send({ ok: false, msg: 'bot not ready' })
    if (req.url.startsWith('/chat')) {
      const since = +(new URL(req.url, 'http://x').searchParams.get('since') || 0)
      return send({ messages: chatLog.filter(m => m.t > since), now: Date.now() })
    }
    if (req.url === '/state') return send(state())
    let a0 = {}
    try { a0 = JSON.parse(body || '{}') } catch (e) {}
    // 'stop' and 'say' bypass the busy lock: the player has to be able to interrupt a long skill and the bot must be
    // able to answer in chat while it works.
    if (busy && a0.skill === 'stop') { const r = await stop(); return send({ ...r, state: state() }) }
    if (busy && a0.skill === 'say') { const r = await say(a0.arg); return send({ ...r, state: state() }) }
    if (busy) return send({ ok: false, msg: 'busy' })
    busy = true
    let out
    try {
      const a = a0
      if (req.url === '/reset') out = await reset(a.x || 0, a.z || 0)
      else out = await withTimeout({ collect: () => collect(a.arg, a.n || 1), craft: () => craft(a.arg), place: () => place(a.arg), smelt: () => smelt(a.arg),
                                     explore: () => explore(a.arg), say: () => say(a.arg), follow: () => follow(a.arg), come: () => come(a.arg),
                                     goto: () => goto(a.arg, a.y, a.z), stop: () => stop(), attack: () => attack(a.arg), look: () => look(a.arg),
                                     tower: () => tower(a.n || a.arg), digdown: () => digdown(a.n || a.arg), toss: () => toss(a.arg, a.n) }[a.skill](),
                       240000, a.skill)
    } catch (e) { out = { ok: false, msg: 'error: ' + e.message } }
    busy = false
    send({ ...out, state: state() })
  })
}).listen(+opt('--http', 3010), '127.0.0.1', () => console.log('skill server on', opt('--http', 3010)))

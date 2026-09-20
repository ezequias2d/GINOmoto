// Quick connect probe: does the bot reach a server, what does it spawn in, and is the server online-mode?
//   node bot/probe.js --host 127.0.0.1 --port 25565 --version 1.21.4 --name GINOmoto
const mineflayer = require('mineflayer')

const argv = process.argv.slice(2)
const opt = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d }

const bot = mineflayer.createBot({
  host: opt('--host', '127.0.0.1'),
  port: +opt('--port', 25565),
  username: opt('--name', 'GINOmoto'),
  version: opt('--version', '1.21.4'),
  auth: opt('--auth', 'offline')
})

bot.once('spawn', () => {
  console.log('spawn ok | version', bot.version, '| pos', bot.entity.position.floored().toString(),
              '| dim', bot.game.dimension, '| health', bot.health, '| food', bot.food)
  const players = Object.keys(bot.players)
  console.log('players online:', players.join(', ') || '(none)')
  bot.chat('oi')
  setTimeout(() => { console.log('quitting'); bot.quit() }, 4000)
})
bot.on('login', () => console.log('login ok, username', bot.username))
bot.on('kicked', r => console.log('KICKED:', JSON.stringify(r)))
bot.on('error', e => console.log('ERROR:', e.message))
bot.on('end', r => console.log('end:', r))

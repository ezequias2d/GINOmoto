#!/usr/bin/env bash
// A fake "player" for testing the chat layer without opening the real client: connects to the server and sends the chat
// orders you pass in. The bot only hears *players*, so this is how an order is simulated from the outside.
//
//   node bot/fake_player.js --name player1 --msg "pega madeira pra mim" --msg "me segue"
const mineflayer = require('mineflayer')

const argv = process.argv.slice(2)
const opt = (k, d) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : d }
const msgs = []
for (let i = 0; i < argv.length; i++) if (argv[i] === '--msg') msgs.push(argv[i + 1])

const bot = mineflayer.createBot({ host: opt('--host', '127.0.0.1'), port: +opt('--port', 25565), username: opt('--name', 'teste'), version: opt('--version', '1.21.4') })
const sleep = ms => new Promise(r => setTimeout(r, ms))

bot.once('spawn', async () => {
  console.log('player', bot.username, 'spawned at', bot.entity.position.floored().toString())
  await sleep(+opt('--pause', 4000))
  for (const m of msgs) {
    bot.chat(m)
    console.log('-> sent:', m)
    await sleep(+opt('--gap', 12000))
  }
  await sleep(2000)
  bot.quit()
})
bot.on('chat', (u, m) => { if (u !== bot.username) console.log(`   [hear] ${u}: ${m}`) })
bot.on('error', e => console.log('error', e.message))
bot.on('kicked', r => console.log('kicked', r))

import asyncio
import websockets

async def f():
    try:
        print("Connecting to ws://127.0.0.1:3000/ws/plume")
        async with websockets.connect('ws://127.0.0.1:3000/ws/plume') as ws:
            print("CONNECTED!")
            await asyncio.sleep(2)
            await ws.send("ping")
            print("Received:", await ws.recv())
    except Exception as e:
        print("Error:", e)

asyncio.run(f())

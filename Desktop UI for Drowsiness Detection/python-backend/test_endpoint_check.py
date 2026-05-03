import warnings; warnings.filterwarnings('ignore')
import sys, logging
logging.getLogger().setLevel(logging.CRITICAL)
import server
c = server.app.test_client()
print('---STATS---')
r = c.get('/api/logs/stats?period=today')
print('status', r.status_code, 'keys', list((r.get_json() or {}).keys()))
d = r.get_json() or {}
print('stats_len', len(d.get('stats', [])))
if d.get('stats'): print('first_stat', d['stats'][0])
print('---SUMMARY---')
r = c.get('/api/logs/summary?period=today')
print('status', r.status_code)
d = r.get_json() or {}
print('summary:', d.get('summary'))
print('---CAMERAS---')
r = c.get('/api/logs/cameras')
print('status', r.status_code)
print((r.get_json() or {}).get('cameras'))
print('---ACTIVE---')
r = c.get('/api/logs/active')
d = r.get_json() or {}
print('status', r.status_code, 'active_students len:', len(d.get('active_students', [])))
print('---EVENTS---')
import sqlite3
conn = sqlite3.connect('drowsiness_logs/events.db')
cid = conn.execute('SELECT DISTINCT camera_id FROM drowsy_events LIMIT 1').fetchone()[0]
r = c.get(f'/api/logs/events/{cid}?period=month')
d = r.get_json() or {}
print(f'camera {cid}: status', r.status_code, 'events len:', len(d.get('events', [])))
if d.get('events'): print('first:', d['events'][0])

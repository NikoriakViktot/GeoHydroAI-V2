# services/terracotta/init_tc_server.py
from terracotta import  update_settings
from terracotta.server import create_app
from flask import jsonify
from config import DRIVER_PATH, TC_PORT, TC_HOST
from flask import Response

update_settings(
    DRIVER_PATH=str(DRIVER_PATH),
    REPROJECTION_METHOD="nearest",   # ок
    RESAMPLING_METHOD="linear"       # ← замість TC_RESAMPLING_METHOD="bilinear"
)

tile_server = create_app()

tile_server.add_url_rule("/health", "health", lambda: jsonify(status="ok"))


@tile_server.route("/apidoc")
def apidoc():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>Terracotta API</title>
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <style> body { margin:0; padding:0; } </style>
      <!-- CDN ReDoc -->
      <script src="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js"></script>
    </head>
    <body>
      <!-- ВАЖЛИВО: відносний шлях, щоб працювало під /tc/ -->
      <redoc spec-url="./swagger.json"></redoc>
    </body>
    </html>
    """
    return Response(html, mimetype="text/html")
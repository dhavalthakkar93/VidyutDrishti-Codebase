#!/bin/bash
# VidyutDrishti — Setup auto-start on Raspberry Pi boot
# Run this script ONCE on the Pi: bash setup_autostart.sh

echo "Setting up VidyutDrishti auto-start..."

# Create systemd service for data_collector
sudo tee /etc/systemd/system/vidyut-collector.service > /dev/null <<'EOF'
[Unit]
Description=VidyutDrishti Data Collector
After=network.target mosquitto.service
Wants=mosquitto.service

[Service]
Type=simple
User=dhaval
WorkingDirectory=/home/dhaval/Desktop/ai_for_bharat
ExecStart=/usr/bin/python3 /home/dhaval/Desktop/ai_for_bharat/data_collector.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Create systemd service for api_server
sudo tee /etc/systemd/system/vidyut-api.service > /dev/null <<'EOF'
[Unit]
Description=VidyutDrishti API Server
After=network.target vidyut-collector.service

[Service]
Type=simple
User=dhaval
WorkingDirectory=/home/dhaval/Desktop/ai_for_bharat
ExecStart=/usr/bin/python3 /home/dhaval/Desktop/ai_for_bharat/api_server.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Enable services
sudo systemctl daemon-reload
sudo systemctl enable vidyut-collector.service
sudo systemctl enable vidyut-api.service

# Start services now
sudo systemctl start vidyut-collector.service
sudo systemctl start vidyut-api.service

echo ""
echo "Done! Services will auto-start on boot."
echo ""
echo "Useful commands:"
echo "  sudo systemctl status vidyut-collector"
echo "  sudo systemctl status vidyut-api"
echo "  sudo journalctl -u vidyut-collector -f  (live logs)"
echo "  sudo journalctl -u vidyut-api -f"
echo "  sudo systemctl restart vidyut-collector"
echo "  sudo systemctl restart vidyut-api"

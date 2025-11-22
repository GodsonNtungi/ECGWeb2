# CardiacTek ECG Analysis V2.0

> Advanced ECG heartbeat classification powered by Machine Learning

## 🎉 What's New in V2

### Major Upgrades
- ✨ **Modern UI/UX** - Complete redesign with responsive layouts
- 🔐 **User Authentication** - Secure login and registration system
- 💾 **Database Integration** - SQLAlchemy ORM with analysis history
- 📊 **Dashboard** - Personal dashboard with statistics and history
- 🎨 **Beautiful Visualizations** - Enhanced ECG graphs with Chart.js
- 🚀 **RESTful API** - Comprehensive API with batch processing
- 🐳 **Docker Support** - Easy deployment with containerization
- 📱 **Responsive Design** - Works on all devices
- 🔒 **Security** - Input validation, rate limiting, and proper error handling

### Technical Improvements
- Modular architecture with Flask Blueprints
- Environment-based configuration
- Comprehensive logging system
- Database migrations support
- Proper error handling and validation
- Dynamic paths (no more hardcoded paths!)
- Production-ready deployment setup

---

## 📋 Features

### 🤖 AI-Powered Analysis
- CatBoost ML model with 99%+ accuracy
- Binary classification: Normal vs Abnormal beats
- Confidence scores for each prediction
- Batch processing support

### 📊 Visualization
- Medical-grade ECG graphs
- Interactive charts and statistics
- Real-time progress indicators
- Downloadable reports

### 👥 User Management
- Secure authentication system
- Personal dashboards
- Analysis history tracking
- Multi-user support

### 🔌 REST API
- `/api/health` - Health check
- `/api/analyze` - Single file analysis
- `/api/batch-analyze` - Batch processing
- JSON responses with comprehensive data

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip
- virtualenv (recommended)

### Installation

1. **Clone the repository**
   ```bash
   cd ECGWeb2
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Initialize database**
   ```bash
   python run.py init-db
   # or
   flask init-db
   ```

6. **Create admin user (optional)**
   ```bash
   flask create-admin
   ```

7. **Run the application**
   ```bash
   python run.py
   ```

8. **Access the application**
   - Web UI: http://localhost:5000
   - API: http://localhost:5000/api/health

---

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Using Docker

```bash
# Build image
docker build -t ecgweb2:latest .

# Run container
docker run -d -p 5000:5000 --name ecgweb2 ecgweb2:latest

# View logs
docker logs -f ecgweb2

# Stop container
docker stop ecgweb2
```

---

## 📁 Project Structure

```
ECGWeb2/
├── app/
│   ├── __init__.py           # App factory
│   ├── models.py             # Database models
│   ├── routes/               # Route blueprints
│   │   ├── main.py          # Main routes
│   │   ├── api.py           # API endpoints
│   │   └── auth.py          # Authentication
│   ├── ml/                   # ML module
│   │   └── model_handler.py # ECG analyzer
│   ├── static/               # Static assets
│   │   ├── css/
│   │   ├── js/
│   │   ├── images/
│   │   └── graphs/          # Generated ECG graphs
│   └── templates/            # Jinja2 templates
├── Models/                   # ML models
│   └── ECGModelsmall.pkl
├── TestData/                 # Sample datasets
├── uploads/                  # Uploaded files (created at runtime)
├── config.py                 # Configuration
├── run.py                    # Entry point
├── requirements.txt          # Dependencies
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose setup
└── README_V2.md             # This file
```

---

## 📊 Usage

### Web Interface

1. **Register/Login**
   - Create an account or login
   - Access your personal dashboard

2. **Upload ECG Data**
   - Navigate to Upload page
   - Drag & drop or browse CSV file
   - Wait for analysis to complete

3. **View Results**
   - See summary statistics
   - Browse individual beat predictions
   - View ECG visualizations
   - Download results as CSV

4. **Dashboard**
   - Track all analyses
   - View statistics
   - Manage history
   - Delete old analyses

### API Usage

#### Analyze Single File

```bash
curl -X POST http://localhost:5000/api/analyze \
  -F "file=@data.csv" \
  -H "Content-Type: multipart/form-data"
```

#### Batch Analysis

```bash
curl -X POST http://localhost:5000/api/batch-analyze \
  -F "files=@data1.csv" \
  -F "files=@data2.csv" \
  -F "files=@data3.csv"
```

#### Health Check

```bash
curl http://localhost:5000/api/health
```

---

## 📝 CSV Format

### Requirements
- Each row = one heartbeat
- 186 data points per row (columns)
- Numeric values only
- Optional header row

### Example
```csv
0.123,0.145,0.167,...,0.234
0.234,0.256,0.278,...,0.345
...
```

### Sample Files
- `TestData/smalldata.csv` - 30 beats
- `TestData/mediumdata.csv` - 1,200 beats
- `TestData/largedata.csv` - 350,000 beats

---

## 🛠️ Development

### Flask CLI Commands

```bash
# Initialize database
flask init-db

# Create admin user
flask create-admin

# Test the model
flask test-model

# Run development server
python run.py

# Run with specific port
PORT=8000 python run.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASK_ENV` | Environment (development/production) | development |
| `SECRET_KEY` | Secret key for sessions | Random |
| `DATABASE_URL` | Database connection string | SQLite |
| `PORT` | Server port | 5000 |

### Configuration Files

- `config.py` - Application configuration
- `.env` - Environment variables (create from `.env.example`)

---

## 🚢 Production Deployment

### Using Gunicorn

```bash
# Install gunicorn (included in requirements.txt)
pip install gunicorn

# Run with 4 workers
gunicorn -w 4 -b 0.0.0.0:8000 run:app

# With timeout for large files
gunicorn -w 4 -b 0.0.0.0:8000 --timeout 300 run:app
```

### Using systemd (Linux)

Create `/etc/systemd/system/ecgweb2.service`:

```ini
[Unit]
Description=ECGWeb2 Application
After=network.target

[Service]
User=www-data
WorkingDirectory=/path/to/ECGWeb2
Environment="PATH=/path/to/ECGWeb2/venv/bin"
ExecStart=/path/to/ECGWeb2/venv/bin/gunicorn -w 4 -b 0.0.0.0:8000 run:app

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable ecgweb2
sudo systemctl start ecgweb2
sudo systemctl status ecgweb2
```

---

## 🔒 Security

- Password hashing with Werkzeug
- CSRF protection
- SQL injection prevention (ORM)
- File upload validation
- Secure session management
- Rate limiting (recommended in production)

---

## 📈 Performance

- Efficient batch processing
- Database query optimization
- Lazy loading for large datasets
- Pagination for results
- Cached model loading

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

- CatBoost ML library
- Flask framework
- Chart.js for visualizations
- Font Awesome for icons

---

## 📞 Support

For issues and questions:
- Email: info@cardiactek.com
- GitHub Issues: Create an issue

---

## 🔮 Roadmap

- [ ] Multi-lead ECG support (12-lead)
- [ ] Real-time WebSocket progress
- [ ] Advanced analytics dashboard
- [ ] PDF report generation
- [ ] Email notifications
- [ ] Model retraining interface
- [ ] Multi-language support
- [ ] Mobile app

---

**Built with ❤️ for better heart health monitoring**

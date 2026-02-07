# Railway Deployment Guide

## Quick Deploy

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/new)

## Manual Deployment Steps

### 1. Prerequisites
- Railway account ([sign up free](https://railway.app))
- GitHub account
- OpenAI API key

### 2. Push to GitHub
```bash
git add -A
git commit -m "Railway deployment configuration"
git push origin master
```

### 3. Deploy on Railway

1. **Create New Project**
   - Go to [railway.app/new](https://railway.app/new)
   - Click "Deploy from GitHub repo"
   - Select your `Psychometricist-AI` repository
   - Railway will auto-detect Python and use the configuration files

2. **Set Environment Variables**
   - In your Railway project dashboard, go to **Variables**
   - Add the following:
     ```
     OPENAI_API_KEY=sk-your-key-here
     ```
   - Optional (for Neo4j graph database):
     ```
     NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
     NEO4J_USERNAME=neo4j
     NEO4J_PASSWORD=your-password
     ```

3. **Deploy**
   - Railway will automatically:
     - Install dependencies via `pip install -e .[web]`
     - Run the app using `python -m web.app`
     - Assign a public URL (e.g., `your-app.up.railway.app`)

### 4. Access Your App

Once deployed, Railway will provide a public URL. Click it or find it in:
- **Settings** → **Domains** → **Generate Domain**

Your app will be live at: `https://your-app-name.up.railway.app`

## Configuration Files

The following files enable Railway deployment:

| File | Purpose |
|------|---------|
| `Procfile` | Defines the startup command for the web process |
| `railway.toml` | Railway-specific build and deployment configuration |
| `.railwayignore` | Excludes unnecessary files from deployment |
| `pyproject.toml` | Python dependencies (Railway auto-detects this) |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | Your OpenAI API key for GPT-5.2 and embeddings |
| `PORT` | ❌ Auto-set | Railway assigns this automatically |
| `NEO4J_URI` | ❌ Optional | Neo4j Aura connection string (uses JSON fallback if not set) |
| `NEO4J_USERNAME` | ❌ Optional | Neo4j username |
| `NEO4J_PASSWORD` | ❌ Optional | Neo4j password |

## Using Neo4j on Railway (Optional)

If you want to use Neo4j instead of the local JSON fallback:

### Option A: Railway's Neo4j Plugin
1. In your Railway project, click **New**
2. Select **Database** → **Neo4j**
3. Railway will provision a Neo4j instance and auto-inject connection variables

### Option B: External Neo4j Aura
1. Sign up at [neo4j.com/aura](https://neo4j.com/aura)
2. Create a free instance
3. Copy credentials and add as Railway environment variables (see above)
4. After deployment, seed the database:
   ```bash
   railway run python -m src.graph.seed
   ```

## Data Persistence

⚠️ **Important**: Railway's ephemeral filesystem means session logs in `data/sessions/` are **lost on redeploy**.

For production, consider:
- Adding a Railway **Volume** mounted at `/app/data`
- Or: Store sessions in a database (PostgreSQL, MongoDB via Railway plugins)
- Or: Use cloud storage (S3, Railway's object storage when available)

## Local Testing Before Deploy

Test the Railway configuration locally:

```bash
# Set PORT environment variable
export PORT=8080  # macOS/Linux
$env:PORT="8080"  # Windows PowerShell

# Run the app
python -m web.app
```

Open `http://localhost:8080` to verify.

## Troubleshooting

### Build Fails
- Check Railway build logs for missing dependencies
- Ensure `pyproject.toml` lists all required packages
- Verify Python version compatibility (requires 3.11+)

### App Crashes on Startup
- Check **Deployments** → **View Logs** in Railway dashboard
- Common issues:
  - Missing `OPENAI_API_KEY` environment variable
  - Port binding errors (ensure using Railway's `PORT` env var)

### "ModuleNotFoundError"
- Ensure `pip install -e .[web]` ran successfully in build logs
- Check that `pyproject.toml` includes the missing module

### Rate Limits / API Errors
- Monitor OpenAI API usage
- Consider implementing rate limiting for public deployments
- Add authentication if exposing publicly

## Monitoring

Railway provides:
- **Metrics**: CPU, memory, network usage
- **Logs**: Real-time stdout/stderr
- **Deployments**: History of all deployments with rollback capability

## Cost Estimation

Railway's free tier includes:
- $5 of usage credit per month
- Sufficient for development and pilot testing
- Production apps may need the Hobby plan ($5/month) or Pro plan ($20/month)

See [Railway's pricing](https://railway.app/pricing) for details.

## Security Notes

For production deployments:
1. **Add authentication** — the current version has no login system
2. **Rate limiting** — prevent API abuse
3. **HTTPS only** — Railway provides this automatically
4. **Environment variables** — never commit `.env` to git
5. **CORS configuration** — restrict allowed origins if embedding

## Next Steps After Deployment

1. Test the full 10-turn interview flow
2. Monitor OpenAI API costs
3. Collect pilot data (N ≥ 5 sessions)
4. Run `python -m src.evaluation.compare` locally to analyze results
5. Consider adding a results dashboard

## Support

- Railway docs: [docs.railway.app](https://docs.railway.app)
- Project issues: [GitHub Issues](https://github.com/Recognifygeneral/Psychometricist-AI/issues)

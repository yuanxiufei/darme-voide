import { Hono } from 'hono'
import { success, badRequest } from '../utils/response.js'
import { listAssetVersions, activateAssetVersion } from '../services/asset-versions.js'

const app = new Hono()

// GET /asset-versions?asset_type=storyboard&asset_id=1
// 资产版本列表（按版本号倒序）
app.get('/', async (c) => {
  try {
    const assetType = c.req.query('asset_type')
    const assetId = c.req.query('asset_id')
    if (!assetType || !assetId) return badRequest(c, 'asset_type and asset_id are required')
    const id = Number(assetId)
    if (!Number.isFinite(id)) return badRequest(c, 'invalid asset_id')

    const versions = listAssetVersions(assetType, id)
    return success(c, { asset_type: assetType, asset_id: id, versions })
  } catch (err: any) {
    return badRequest(c, err?.message || 'list asset versions failed')
  }
})

// POST /asset-versions/:id/activate
// 回滚：把指定版本置为 current，并将 asset_url 写回主表字段
app.post('/:id/activate', async (c) => {
  try {
    const id = Number(c.req.param('id'))
    if (!Number.isFinite(id)) return badRequest(c, 'invalid version id')

    const result = activateAssetVersion(id)
    if (!result.ok) return badRequest(c, result.error || 'activate failed')
    return success(c, { activated: result.row })
  } catch (err: any) {
    return badRequest(c, err?.message || 'activate asset version failed')
  }
})

export default app

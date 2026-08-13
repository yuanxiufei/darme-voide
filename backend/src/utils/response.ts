import type { Context } from 'hono'

export function success(c: Context, data: any = null) {
  return c.json({ code: 200, data, message: 'success' })
}

export function created(c: Context, data: any = null) {
  return c.json({ code: 201, data, message: 'created' }, 201)
}

export function badRequest(c: Context, message = 'bad request') {
  return c.json({ code: 400, message }, 400)
}

export function notFound(c: Context, message = 'not found') {
  return c.json({ code: 404, message }, 404)
}

export function serverError(c: Context, message = 'internal error') {
  return c.json({ code: 500, message }, 500)
}

export function now() {
  return new Date().toISOString()
}

/** 安全解析路由参数 id，NaN 时返回 null */
export function parseParamId(c: Context, paramName = 'id'): number | null {
  const raw = c.req.param(paramName)
  const id = Number(raw)
  return Number.isFinite(id) && id > 0 ? id : null
}

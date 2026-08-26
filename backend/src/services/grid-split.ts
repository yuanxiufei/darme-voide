import sharp from 'sharp'
import fs from 'fs'
import path from 'path'
import { now } from '../utils/response.js'
import { getAbsolutePath } from '../utils/storage.js'

interface SplitResult {
  index: number
  localPath: string
}

export async function splitGridImage(
  imagePath: string,
  rows: number,
  cols: number,
): Promise<SplitResult[]> {
  if (!Number.isInteger(rows) || rows < 1 || !Number.isInteger(cols) || cols < 1) {
    throw new Error(`Invalid grid dimensions: rows=${rows}, cols=${cols}. Both must be positive integers.`)
  }
  const absPath = imagePath.startsWith('/')
    ? imagePath
    : getAbsolutePath(imagePath)

  const image = sharp(absPath)
  const meta = await image.metadata()
  if (!meta.width || !meta.height) throw new Error('Cannot read image dimensions')

  const cellW = Math.floor(meta.width / cols)
  const cellH = Math.floor(meta.height / rows)

  const outDir = getAbsolutePath('grid-cells')
  fs.mkdirSync(outDir, { recursive: true })

  const results: SplitResult[] = []
  const ts = Date.now()

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const index = r * cols + c
      const fileName = `cell_${ts}_${index}.png`
      const outPath = path.join(outDir, fileName)

      await sharp(absPath)
        .extract({ left: c * cellW, top: r * cellH, width: cellW, height: cellH })
        .toFile(outPath)

      results.push({
        index,
        localPath: `static/grid-cells/${fileName}`,
      })
    }
  }

  return results
}

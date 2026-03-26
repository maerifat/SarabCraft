import { SARABCRAFT_R1_NAME } from '../../constants/threat'

export { SARABCRAFT_R1_NAME }

export const SARABCRAFT_R1_STRATEGY_OPTIONS = [
  { value: 'tile_shuffle', label: 'Tile Shuffle' },
  { value: 'progressive', label: 'Progressive Bank' },
  { value: 'self_aug_bank', label: 'Self-Augmented Bank' },
  { value: 'self_mix', label: 'Self Mix' },
]

export function isSarabCraftR1(attackName) {
  return attackName === SARABCRAFT_R1_NAME
}

export function buildAttackParamPayload(attackName, attackParams, getParam) {
  const params = {}

  Object.keys(attackParams || {}).forEach((key) => {
    if (isSarabCraftR1(attackName) && ['r1_multi_image_strategy', 'r1_multi_image_count'].includes(key)) return
    params[key] = getParam(key)
  })

  if (isSarabCraftR1(attackName)) {
    const multiImage = Boolean(getParam('r1_multi_image'))
    params.r1_multi_image = multiImage
    if (multiImage) {
      params.r1_multi_image_strategy = getParam('r1_multi_image_strategy') || 'tile_shuffle'
      params.r1_multi_image_count = Number(getParam('r1_multi_image_count') || 10)
    }
  }

  return params
}

export function sarabCraftR1StrategyHint(value) {
  switch (value) {
    case 'tile_shuffle':
      return 'Create a pseudo-batch from the current image for the strongest default multi-image transfer setup.'
    case 'progressive':
      return 'Start with broader reference diversity, then gradually refocus on source-image features.'
    case 'self_aug_bank':
      return 'Build the reference bank from augmented copies of the source image for a balanced transfer study.'
    case 'self_mix':
      return 'Use the lightest self-mix variant when you want a cheaper multi-image style run.'
    default:
      return ''
  }
}

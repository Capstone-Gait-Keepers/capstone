import { reactive } from 'vue'
import type { User } from '@/types'

export const store = reactive<{
  user: User | null;
}>({
  user: {
    email: "dan@raymond.ch",
    password: "test",
    sensor_id: 39,
  }, // TODO: Default should be null
})
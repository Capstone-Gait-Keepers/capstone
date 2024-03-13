import { reactive } from 'vue'
import type { User } from '@/types'

export default reactive<{
  user: User | null;
}>({
  user: {
    name: "Marjorie",
    email: "marjorie.smith@gmail.com",
    password: "test",
    sensorid: 39,
  }, // TODO: Default should be null
});

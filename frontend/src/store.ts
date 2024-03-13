import { reactive } from 'vue'
import type { User } from '@/types'

export default reactive<{
  user: User | null;
  connected_users: string[];
  viewed_categories: Record<string, boolean>;
}>({
  user: {
    name: "Marjorie",
    email: "marjorie.smith@gmail.com",
    password: "thisisapassword",
    sensor_id: 39,
  }, // TODO: Default should be null
  connected_users: ['dan@raymond.ch'],
  viewed_categories: {
    Balance: true,
    Neurodegenerative: true,
    Dementia: true,
  }
});

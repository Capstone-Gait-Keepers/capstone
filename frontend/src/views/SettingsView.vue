<template>
  <BasePage v-if="store.user">
    <h1>Account Settings</h1>
    <h2>Account Information</h2>
    <FormInput label="Name" type="text" v-model="store.user.name" />
    <FormInput label="Email" type="email" v-model="store.user.email" />
    <FormInput label="Password" type="password" v-model="store.user.password" />
    <FormInput label="Sensor ID" type="text" v-model="store.user.sensorid" />
    <!-- <FormInput label="Notifications" type="text" v-model="store.user.notifications" /> -->
    <h2>Metrics</h2>
    <p>Choose what information to view on the Home Insights page. If you don’t want to see a specific group of insights or recommendations simply turn the toggle off. The system will still collect the data it just won’t display it to you so you can turn them back on at any time if you decide you want to see the insights.</p>
    <Toggle label="Balance indicators" />
    <Toggle label="Neurodegenerative disease indicators" />
    <Toggle label="Dementia Indicators" />
    <h2>Connected Users</h2>
    <i>Adding family members will allow them to view your individual insights from their own account. Once you add a family member they will receive an email inviting them to create an account and link your data to it. </i>
    <!-- <FormInput label="Name" type="text" /> -->
  </BasePage>
</template>

<script setup lang="ts">
import BasePage from '@/components/BasePage.vue';
import FormInput from '@/components/FormInput.vue';
import Toggle from '@vueform/toggle';
import store from '@/store';

console.log(store.user?.name);
console.log(store.user?.email);
</script>